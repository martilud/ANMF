import numpy as np
from utils import *
from copy import deepcopy

class NMF:

    def __init__(self, d = None, ds = None, tau_W = 1.0, tau_A = 0.1, tau_S = 0.0,
        omega = None, omegas = None, W = None, prob = "std", init = "exem", update_H = False,
        epochs = 50, test_epochs = 50, warm_start_epochs = 0,
        batch_size = None, batch_size_adv = None, batch_size_sup = None,
        true_sample = "std", normalize = False,
        mu_W = 1e-6, mu_H = 1e-6):
        """
        Class for Non-Negative Matrix Factorization for source separation. 
        Decomposes Non-negative data stored columnwise in and m times n array U as U approx WH,
        where W is m times d and H is d times n.

        Class has several fit functions for different variants of NMF.
        
        Parameters:
        -----------
        d (int): number of basis vectors in W.
        ds (list): number of basis vectors in W, for DNMF and D+MDNMF.
        tau_W (float): Weak Supervision Weight.
        tau_A (float): Adversarial weight.
        tau_S (float): Strong supervision weight.
        omega (array): Weights of adversarial data 
        omegas (list): Weghts of adversarial data, for D+MDNMF.
        mu_W (float): regularization parameter for W.
        mu_H (float,list): regularization parameter for H.
        prob (str): type of NMF. "std" for standard, "adv" for MDNMF, 
                    "sup" for DNMF and "full" for D+MDNMF.
        init (str): initialization method. "random" for random initialization and 
                    "exem" for exemplar-based initialization.
        epochs (int): number of training epochs.
        test_epochs (int): number of epochs used in testing.
        warm_start_epochs (int): number of warm-start epochs before the main training.
        batch_size (int): size of mini-batches for true data term.
        batch_size_adv (int): size of mini-batches for adversarial data term
        batch_size_sup (int): size of mini-batches for strong supervision data term.
        true_sample (str): which dataset to fully sample. Data is shuffled and new epoch is started
                            when all data in this dataset has been passed through. 
                           "std" for true data, "adv" for adversarial data, 
                           "sup" for supervised data.
        normalize (bool): whether or not to normalize the columns of W.
        update_H (bool): whether or not to do batchwise updates of H during training. 
        """
        self.mu_W = mu_W
        self.prob = prob
        self.init = init
        self.epochs = epochs
        self.test_epochs = test_epochs
        self.warm_start_epochs = warm_start_epochs
        self.d = d
        self.ds = ds
        self.batch_size = batch_size
        self.batch_size_adv = batch_size_adv
        self.batch_size_sup = batch_size_sup
        self.true_sample = true_sample
        self.omega = omega
        self.omegas = omegas
        self.tau_W = tau_W
        self.tau_A = tau_A
        self.tau_S = tau_S
        self.normalize = normalize
        self.update_H = update_H

        self.set_mu_H(mu_H)

        # Copy W if it is given
        self.W = W

        # Initialize H, H_adv and H_sup
        self.H = None
        self.H_adv = None
        self.H_sup = None

        # Lengths
        self.M = None
        self.N = None
        self.N_adv = None
        self.N_sup = None

        # Number of sources
        self.S = None

        # Calculate what parts of W and H_sup correspond
        # to what source in relevant cases
        if self.ds is not None:
            self.source_ids = []
            for i in range(len(self.ds)):
                self.source_ids.append(np.arange(sum(self.ds[:i]), sum(self.ds[:i+1])))

    def set_mu_H(self,mu_H):
        if isinstance(mu_H,(int,float)):
            self.mu_H = np.array([mu_H]*self.d)
        elif self.ds is not None:
            if type(mu_H) == list:
                if len(mu_H) == len(self.ds):
                    self.mu_H = np.array([mu_H[i] for i in range(len(self.ds)) for _ in range(self.ds[i])])
        else:
            self.mu_H = np.array(mu_H)

    def std_W_update(self,U,W,H, HHT = None):
        """
        Update rule for W for standard NMF and discriminative NMF
        """
        if HHT is None:
            HHT = np.dot(H, H.T)
        invN = 1.0/U.shape[1]

        return np.dot(U, H.T) * invN / (np.dot(W, HHT) * invN + self.mu_W)

    def adv_W_update(self, U, U_adv, W, H, H_adv, UHT = None, U_advH_advT = None, HHT = None, H_advH_advT = None):
        """
        Update rule for adversarial NMF
        """
        if UHT is None:
            UHT = np.dot(U,H.T)
        if U_advH_advT is None:
            U_advH_advT = np.dot(U_adv,H_adv.T)
        if HHT is None:
            HHT = np.dot(H,H.T)
        if H_advH_advT is None:
            H_advH_advT = np.dot(H_adv,H_adv.T)

        invN = 1.0/U.shape[1]
        tau_invN_adv = self.tau_A/U_adv.shape[1]

        return (np.dot(W,H_advH_advT) * tau_invN_adv + UHT * invN)/(np.dot(W, HHT)*invN + U_advH_advT * tau_invN_adv + self.mu_W)

    def std_W_semi_update(self,V,W,H, HHT = None):
        """
        Update rule for W in semi-supervised data setting where data from the last source is missing
        """
        invN_mix = 1.0/V.shape[1]
        if HHT is None:
            HHT = np.dot(H, H[self.source_ids[-1],:].T)
        return np.dot(V,H[self.source_ids[-1],:].T) * invN_mix/(np.dot(W, HHT) * invN_mix + self.mu_W)

    # Semi supervised + adversarial update, never used
    #def adv_W_semi_update(self,V,U_adv, W, H, H_adv, VHT = None, U_advH_advT = None, HHT = None, HH_advT = None):
    #    """
    #    Update rule for W for standard NMF in semi-supervised data setting 
    #    """
    #    invN_mix = 1.0/V.shape[1]
    #    invN_adv = 1.0/U_adv.shape[1]
    #    if VHT is None:
    #        VHT = np.dot(V, H[self.source_ids[-1],:].T)
    #    if U_advH_advT is None:
    #        U_advH_advT = np.dot(U_adv, H_adv.T)
    #    if HHT is None:
    #        HHT = np.dot(H, H[self.source_ids[-1],:].T)
    #    if HH_advT is None:
    #        HH_advT = np.dot(H_adv, H_adv.T)

    #    return (np.dot(V,H[self.source_ids[-1],:].T) * invN_mix + np.dot(W[:,self.source_ids[-1]],HH_advT) * invN_adv)/(np.dot(W, HHT) * invN_mix + U_advH_advT * invN_adv + self.mu_W)

    def H_update(self,U,W,H, omega = None, WtU = None, WtW = None):
        """
        Update rule for H for standard NMF and discriminative NMF
        """
        if WtU is None:
            WtU = np.dot(W.T, U)
        if WtW is None:
            WtW = np.dot(W.T, W)

        if omega is not None:
            return WtU/(np.dot(WtW, H) + np.outer(self.mu_H, omega))
        else:
            return WtU/(np.dot(WtW, H) + self.mu_H[:,np.newaxis])

    def H_update_source(self,U,W,H,source, omega = None, WtU = None, WtW = None):
        """
        Update rule for H for standard NMF and discriminative NMF
        """
        if WtU is None:
            WtU = np.dot(W.T, U)
        if WtW is None:
            WtW = np.dot(W.T, W)
        if omega is not None:
            return WtU /(np.dot(WtW, H) + np.outer(self.mu_H[self.source_ids[source]], omega))
        else: 
            return WtU /(np.dot(WtW, H) + self.mu_H[self.source_ids[source],np.newaxis])

    def H_semi_update(self,V,W,H,source,WitW = None):
        """
        Update rule for H in semi-supervised setting where data from the last source is missing
        """
        
        if WitW is None:
            WitW = np.dot(W[:,self.source_ids[source]].T, W)
        
        return (np.dot(W[:,self.source_ids[source]].T, V))/(np.dot(WitW, H) + self.mu_H[self.source_ids[source],np.newaxis])


    def initializeWH(self, U = None, U_adv = None, U_sup = None, V_sup = None, prob = "std"):
        """
        Initialize W and H for fitting

        TO DO: Move this into initW and initH
        """

        if self.W is None:
            if self.init == "rand":
                self.W = np.random.uniform(0,1,(self.M,self.d))
            elif self.init == "exem":
                if prob == "std" or prob == "adv" or prob == "exem":
                    self.W = U[:,np.random.choice(self.N, size = self.d, replace = False)]
                else:
                    self.W = np.zeros((self.M,self.d))
                    for j in range(len(self.ds)):
                        if U is not None:
                            self.W[:,self.source_ids[j]] = U[j][:,np.random.choice(self.N, size = self.ds[j], replace = False)]
                        else:
                            self.W[:,self.source_ids[j]] = U_sup[j][:,np.random.choice(self.N_sup, size = self.ds[j], replace = False)]

        if self.normalize == True:
            norms = np.linalg.norm(self.W, axis = 0)
            self.W = self.W/(norms + 1e-10)

        if self.init == "rand":
            if prob != "sup":
                self.H = np.random.uniform(0,1,(self.d,self.N))
                if prob == "adv" or prob == "full":
                    self.H_adv = np.random.uniform(0,1,(self.d,self.N_adv))
            if prob == "sup" or prob == "full":
                self.H_sup = np.random.uniform(0,1,(self.d,self.N_sup))

        elif self.init == "exem":
            if prob == "std" or prob == "adv" or prob == "exem":
                self.H = self.transform(U)
                if prob == "adv":
                    self.H_adv = self.transform(U_adv)
            elif prob == "full":
                self.H = np.zeros((self.d, self.N))
                self.H_adv = np.zeros((self.d, self.N_adv))
                for j in range(len(self.ds)):
                    WtW = np.dot(self.W[:,self.source_ids[j]].T, self.W[:,self.source_ids[j]])
                    self.H[self.source_ids[j],:] = self.transform_source(U[j], source = j, WtW = WtW, WtU = np.dot(self.W[:,self.source_ids[j]].T, U[j])) 
                    self.H_adv[self.source_ids[j],:] = self.transform_source(U_adv[j], source = j, WtW = WtW, WtU = np.dot(self.W[:,self.source_ids[j]].T, U_adv[j])) 
            if prob == "sup" or prob == "full":
                self.H_sup = self.transform(V_sup)
    
    def initW(self,U = None, U_adv = None, U_sup = None, V_sup = None, prob = "std"):
        """
        Initialize W 

        TO DO: Remove prob argument, use class attribute
        """
        if self.W is None:
            if self.init == "rand":
                self.W = np.random.uniform(0,1,(self.M,self.d))
            elif self.init == "exem":
                if prob == "std" or prob == "adv" or prob == "exem":
                    self.W = U[:,np.random.choice(self.N, size = self.d, replace = False)]
                else:
                    self.W = np.zeros((self.M,self.d))
                    for j in range(len(self.ds)):
                        if U is not None:
                            self.W[:,self.source_ids[j]] = U[j][:,np.random.choice(self.N, size = self.ds[j], replace = False)]
                        else:
                            self.W[:,self.source_ids[j]] = U_sup[j][:,np.random.choice(self.N_sup, size = self.ds[j], replace = False)]

        if self.normalize == True:
            norms = np.linalg.norm(self.W, axis = 0)
            self.W = self.W/(norms + 1e-10)
    
    def initH(self,U = None, U_adv = None, V = None, U_sup = None, V_sup = None, prob = "std"):
        """
        Initialize latent variables 

        TO DO: Remove prob argument, use class attribute
        """

        if self.init == "rand":
            if prob != "sup":
                self.H = np.random.uniform(0,1,(self.d,self.N))
                if prob == "adv" or prob == "full" or prob == "semi_adv":
                    self.H_adv = np.random.uniform(0,1,(self.d,self.N_adv))
            if prob == "sup" or prob == "full":
                self.H_sup = np.random.uniform(0,1,(self.d,self.N_sup))

        elif self.init == "exem":
            assert self.W is not None, "Exemplar based initialization requires W not None"
            if prob == "std" or prob == "adv" or prob == "exem":
                self.H = self.transform(U)
                if prob == "adv":
                    self.H_adv = self.transform(U_adv)
            elif prob == "full":
                self.H = np.zeros((self.d, self.N))
                self.H_adv = np.zeros((self.d, self.N_adv))
                for j in range(len(self.ds)):
                    WtW = np.dot(self.W[:,self.source_ids[j]].T, self.W[:,self.source_ids[j]])
                    self.H[self.source_ids[j],:] = self.transform(U[j], WtW = WtW, WtU = np.dot(self.W[:,self.source_ids[j]].T, U[j])) 
                    self.H_adv[self.source_ids[j],:] = self.transform(U_adv[j], WtW = WtW, WtU = np.dot(self.W[:,self.source_ids[j]].T, U_adv[j])) 
            elif prob == "semi" or prob == "semi_adv":
                self.H = self.transform(V)
                if prob == "semi_adv":
                    self.H_adv = np.random.uniform(0,1,(self.ds[-1],self.N_adv))
            if prob == "sup" or prob == "full":
                self.H_sup = self.transform(V_sup)

    def std_loss(self,U, WH):
        """
        Loss for standard NMF
        """
        return 1/U.shape[1] * np.linalg.norm(U - WH, 'fro')**2
    
    def adv_loss(self,U,U_adv,WH,WH_adv):
        """
        Loss for ANMF

        TO DO: Double check if tau_A is multiplied redundantly here, most likely it can be removed.
        """
        return self.tau_W/U.shape[1] * np.linalg.norm(U - WH, 'fro')**2 - self.tau_A/U_adv.shape[1] * np.linalg.norm(U_adv - WH_adv, 'fro')**2

    def fit_std(self, U, conv = False):
        """
        Fits standard NMF by solving

        min_{W \ge 0} 1/N \|U - WH(U,W)\|_F^2
        where H(U,W) = arg min_{H \ge 0} \|U - WH\|_F^2
        """

        self.M = U.shape[0]
        self.N = U.shape[1]

        # Calculate Number of Batches
        if self.batch_size == None:
            self.batch_size = U.shape[1]
        if self.prob == "sup" and self.batch_size_sup is not None:
            self.batch_size = self.batch_size_sup

        N_batches = self.N//self.batch_size 

        # Initialize if nothing exists
        if self.W is None or self.H is None:
            self.initializeWH(U = U, prob = "std")

        # List of ids that will be shuffled
        ids = np.arange(0,self.N)


        # Define updates and loss func, leftover from old code
        W_update = self.std_W_update
        H_update = self.H_update

        # Define array we will need for convergence
        if conv:
            loss_func = self.std_loss
            loss_std = np.zeros((self.epochs + 1))
            WH = np.dot(self.W,self.H)
            loss_std[0] = loss_func(U,WH)

        for i in range(self.epochs):
            # Shuffle ids, U and H
            np.random.shuffle(ids)

            for b in range(N_batches):
                
                # Technically do not need the np.arange here
                batch_ids = ids[np.arange(b*self.batch_size,(b+1)*self.batch_size)%self.N]

                # Makes copies of the shuffled data 
                U_batch = U[:,batch_ids]
                H_batch = self.H[:,batch_ids]

                # Update W for each batch
                self.W *= W_update(U_batch, self.W, H_batch)

                # Update H batchwise
                if self.update_H:

                    H_batch *= H_update(U_batch,self.W,H_batch)

                    self.H[:,batch_ids] = H_batch

            # Update H if not done batchwise
            if not self.update_H:
                self.H *= H_update(U,self.W,self.H)
            
            # Normalize columns of W
            if self.normalize:
                norms = np.linalg.norm(self.W, axis = 0)
                self.W /=(norms + 1e-10)
                self.H *= norms[:,np.newaxis]

            # Storing convergence
            if conv:
                WH = np.dot(self.W,self.H)
                loss_std[i+1] = loss_func(U, WH)

            
        if conv:
            return loss_std

    def fit_adv(self, U, U_adv, conv = False):
        """
        Fits adversarial NMF by solving

        min_{W \ge 0} 1/N \|U - WH(U,W)\|_F^2 - tau_A/\hat{N} \|U - WH(\hat{U},W)\|_F^2 
        where H(U,W) = arg min_{H \ge 0} \|U - WH\|_F^2

        Here U is true data and \hat{U} is adversarial data.
        """
        self.M = U.shape[0]
        self.N = U.shape[1]

        assert self.M == U_adv.shape[0], f"U has first axis {self.M} and U_adv has first axis {U_adv.shape[0]} which does not match."

        self.N_adv = U_adv.shape[1]

        # Calculate umber of Batches
        if self.batch_size == None:
            self.batch_size = self.N
        if self.batch_size_adv == None:
            self.batch_size_adv = self.N_adv

        N_batches = self.N//self.batch_size
        N_batches_adv = self.N_adv//self.batch_size_adv  
        
        # Used for inner epoch loop
        if self.true_sample == "std":
            N_batches = N_batches
        elif self.true_sample == "adv":
            N_batches = N_batches_adv
        else:
            N_batches = np.minimum(N_batches, N_batches_adv)

        if self.W is None or self.H is None or self.H_adv is None:
            self.initializeWH(U = U, U_adv = U_adv, prob = "adv")

        # List of ids that will be shuffled
        ids = np.arange(0,self.N)
        ids_adv = np.arange(0,self.N_adv)

        # Set update
        W_update = self.adv_W_update
        H_update = self.H_update
        
        if conv:
            loss_func = self.adv_loss
            loss_adv = np.zeros((self.epochs + 1))
            WH = np.dot(self.W,self.H)
            WH_adv = np.dot(self.W,self.H_adv)
            loss_adv[0] = loss_func(U,U_adv, WH, WH_adv)
            

        for i in range(self.epochs):
            # Shuffle ids, U and H
            np.random.shuffle(ids)
            np.random.shuffle(ids_adv)

            for b in range(N_batches):

                batch_ids = ids[np.arange(b*self.batch_size,(b+1)*self.batch_size)%self.N]
                batch_ids_adv = ids_adv[np.arange(b*self.batch_size_adv,(b+1)*self.batch_size_adv)%self.N_adv]

                # Makes copies of the shuffled data
                U_batch = U[:,batch_ids]
                H_batch = self.H[:,batch_ids]
                U_adv_batch = U_adv[:,batch_ids_adv]
                H_adv_batch = self.H_adv[:,batch_ids_adv]

                # Update W for each batch
                self.W *= W_update(U_batch, U_adv_batch, self.W, H_batch, H_adv_batch)

                # Update H for each batch
                if self.update_H:

                    WtW = np.dot(self.W.T, self.W)
                    H_batch *= H_update(U_batch,self.W,H_batch, WtW = WtW)
                    H_adv_batch *= H_update(U_adv_batch,self.W,H_adv_batch, omega = self.omega[batch_ids_adv], WtW = WtW)

                    self.H[:,batch_ids] = H_batch
                    self.H_adv[:,batch_ids_adv] = H_adv_batch

                
            if not self.update_H: 
                WtW = np.dot(self.W.T, self.W)
                self.H *= H_update(U,self.W,self.H, WtW = WtW)
                self.H_adv *= H_update(U_adv,self.W,self.H_adv, omega = self.omega, WtW = WtW)

            if self.normalize:
                norms = np.linalg.norm(self.W, axis = 0)
                self.W /= (norms + 1e-10)

                # Assumes true_sample is std !!!!!
                self.H *= norms[:,np.newaxis]
                self.H_adv *= norms[:,np.newaxis]

            if conv:
                WH = np.dot(self.W,self.H)
                WH_adv = np.dot(self.W,self.H_adv)
                loss_adv[i+1] = loss_func(U,U_adv, WH, WH_adv)
                
        if conv:
            return loss_adv

    def fit_sup(self,U_sup,V_sup,conv = False):
        """
        Solves:

        \min_{W \ge 0} \sum_i^{S} \|U_i - W_i H_i\|_F^2
        s.t H = \argmin_{H} \|V - WH\|_F^2,
        where W = [W_1,...W_S] and H = [H_1,... W_S] are concatenated versions of W and H.
        Technically solves this on multi-objective form.

        U should be input as a list of (M,N) arrays

        TO DO: Make it handle (M,S,N) arrays as well
        """
        assert self.ds is not None and self.d is not None, "Supervised fitting needs both d and ds"
        assert len(U_sup) == len(self.ds), "U_sup and self.ds shapes do not match"
        #self.Ms = np.zeros(len(U), dtype = int)


        self.N_sup = V_sup.shape[1] 
        self.M = V_sup.shape[0]
        self.S = len(U_sup)

        
        # Calculate Number of Batches
        if self.batch_size_sup is None:
            self.batch_size_sup = self.N_sup
        self.N_batches = self.N_sup//self.batch_size_sup 

        if self.W is None or self.H is None:
            self.initializeWH(U_sup = U_sup, V_sup = V_sup, prob = "sup")

        ids = np.arange(0,self.N_sup)

        W_update = self.std_W_update
        H_update = self.H_update


        if conv:
            loss_func = self.std_loss
            loss_sup = np.zeros((self.S, self.epochs + 1))
            for j in range(self.S):
                WH_sup = np.dot(self.W[:,self.source_ids[j]], self.H_sup[self.source_ids[j],:])
                loss_sup[j,0] = loss_func(U_sup[j], WH_sup)
        
        for i in range(self.epochs):

            # Shuffle data
            np.random.shuffle(ids)

            # Iterate over each source

            for b in range(self.N_batches):

                batch_ids = ids[np.arange(b*self.batch_size_sup,(b+1)*self.batch_size_sup)%self.N_sup]

                for j in range(self.S):
                    
                    U_sup_batch = U_sup[j][:,batch_ids]
                    H_sup_batch = self.H_sup[self.source_ids[j]][:,batch_ids]
                    
                    # Update W
                    self.W[:,self.source_ids[j]] *= W_update(U_sup_batch, self.W[:,self.source_ids[j]], H_sup_batch)

                    if self.update_H:
                        V_sup_batch = V_sup[:,batch_ids]
                        self.H_sup[:,batch_ids] *= H_update(V_sup_batch, self.W, self.H_sup[:,batch_ids])
                
            if not self.update_H:
                self.H_sup *= H_update(V_sup,self.W,self.H_sup) 

            if self.normalize:
                norms = np.linalg.norm(self.W, axis = 0)
                self.W = self.W/(norms + 1e-10)
                self.H_sup = norms[:,np.newaxis] * self.H_sup
            
            if conv:
                for j in range(self.S):
                    WH_sup = np.dot(self.W[:,self.source_ids[j]], self.H_sup[self.source_ids[j],:])
                    loss_sup[j,i+1] = loss_func(U_sup[j], WH_sup)
        if conv:
            return loss_sup

    def fit_exem(self, U = None):
        """
        Fits Exemplar-based NMF. 
        """

        self.N = U.shape[1]

        # Fitting of exemplar-based NMF is handled by the initialization function
        self.initW(U = U)

    
    def fit_full(self, U, U_adv, U_sup, V_sup, conv = False):
        """
        Fit function that can handle fitting for fitting D+ANMF which includes weak supervision data, adversarial data
        and strong supervision data. 

        TO DO: Implementation that handles unbalanced data. Rewrite so it does not make redundant copies of data.

        """

        # Store sizes
        self.N = U[0].shape[1]
        self.N_adv = U_adv[0].shape[1]
        self.N_sup = V_sup.shape[1]
        self.M = V_sup.shape[0]
        self.S = len(U_sup)

        # Initialize W and the different latent variables
        self.initializeWH(U = U, U_adv = U_adv, U_sup = U_sup, V_sup = V_sup, prob = "full")

        if self.batch_size == None:
            self.batch_size = self.N
        if self.batch_size_adv == None:
            self.batch_size_adv = self.N_adv
        if self.batch_size_sup == None:
            self.batch_size_sup = self.N_sup

        # Calculate number of batches
        N_batches = self.N//self.batch_size
        N_batches_adv = self.N_adv//self.batch_size_adv 
        N_batches_sup = self.N_sup//self.batch_size_sup

        inv_N = self.tau_W/self.batch_size
        inv_N_adv = self.tau_A/self.batch_size_adv
        inv_N_sup = self.tau_S/self.batch_size_sup
        
        # Select which data to true sample
        if self.true_sample == "std":
            N_batches = N_batches
        elif self.true_sample == "adv":
            N_batches = N_batches_adv
        elif self.true_sample == "sup":
            N_batches = N_batches_sup
        else:
            N_batches = np.minimum(N_batches, N_batches_adv, N_batches_sup)
        

        ids = np.arange(0,self.N)
        ids_adv = np.arange(0,self.N_adv)
        ids_sup = np.arange(0,self.N_sup)

        if conv:
            loss_func = lambda U, U_adv, U_sup, WH, WH_adv, WH_sup : self.adv_loss(U, U_adv, WH, WH_adv) + self.tau_S * self.std_loss(U_sup, WH_sup)
            loss_full = np.zeros((self.S, self.epochs + 1))
            for j in range(self.S):
                WH = np.dot(self.W[:,self.source_ids[j]], self.H[self.source_ids[j],:]) 
                WH_adv = np.dot(self.W[:,self.source_ids[j]], self.H_adv[self.source_ids[j],:]) 
                WH_sup = np.dot(self.W[:,self.source_ids[j]], self.H_sup[self.source_ids[j],:])
                loss_full[j,0] = loss_func(U[j], U_adv[j], U_sup[j], WH, WH_adv, WH_sup)


        # Start main loop
        for i in range(self.epochs):
            # Shuffle data

            np.random.shuffle(ids)
            np.random.shuffle(ids_adv)
            np.random.shuffle(ids_sup)

            # Update H_sup
            self.H_sup *= self.H_update(V_sup,self.W,self.H_sup) 
            
            # Update W for each source
            for j in range(self.S):

                # Split data into batches
                U_b = np.array_split(U[j][:,ids], N_batches, axis = 1)
                H_b = np.array_split(self.H[self.source_ids[j]][:,ids], N_batches, axis = 1)
                U_adv_b = np.array_split(U_adv[j][:,ids_adv], N_batches_adv, axis = 1)
                H_adv_b = np.array_split(self.H_adv[self.source_ids[j]][:,ids_adv], N_batches_adv, axis = 1) 
                U_sup_b = np.array_split(U_sup[j][:,ids_sup], N_batches_sup, axis = 1)
                H_sup_b = np.array_split(self.H_sup[self.source_ids[j]][:,ids_sup], N_batches_sup, axis = 1)
                
                for b in range(N_batches):
                    # Calculate terms needed for W update
                    top = np.dot(U_b[b%N_batches], H_b[b%N_batches].T) * inv_N
                    top += np.dot(self.W[:,self.source_ids[j]], np.dot(H_adv_b[b%N_batches_adv], H_adv_b[b%N_batches_adv].T)) * inv_N_adv
                    top += np.dot(U_sup_b[b%N_batches_sup], H_sup_b[b%N_batches_sup].T) * inv_N_sup

                    bot = np.dot(self.W[:,self.source_ids[j]], np.dot(H_b[b%N_batches], H_b[b%N_batches].T)) * inv_N
                    bot += np.dot(U_adv_b[b%N_batches_adv], H_adv_b[b%N_batches_adv].T) * inv_N_adv
                    bot += np.dot(self.W[:,self.source_ids[j]], np.dot(H_sup_b[b%N_batches_sup], H_sup_b[b%N_batches_sup].T)) * inv_N_sup

                    # Calculate W update
                    self.W[:,self.source_ids[j]] *= (top)/(bot + self.mu_W)

                WtW = np.dot(self.W[:,self.source_ids[j]].T, self.W[:,self.source_ids[j]])
                self.H[self.source_ids[j],:] *= self.H_update_source(U[j],self.W[:,self.source_ids[j]],self.H[self.source_ids[j],:], source = j, WtW = WtW) 
                self.H_adv[self.source_ids[j],:] *= self.H_update_source(U_adv[j], self.W[:,self.source_ids[j]],self.H_adv[self.source_ids[j],:], omega = self.omegas[j] if self.omegas is not None else None, source = j, WtW = WtW)  


            if self.normalize:
                norms = np.linalg.norm(self.W, axis = 0)
                self.W /= (norms + 1e-10)
                self.H *= norms[:,np.newaxis]
                self.H_adv *= norms[:,np.newaxis]
                self.H_sup *= norms[:,np.newaxis]

            if conv:
                for j in range(self.S):
                    WH = np.dot(self.W[:,self.source_ids[j]], self.H[self.source_ids[j],:]) 
                    WH_adv = np.dot(self.W[:,self.source_ids[j]], self.H_adv[self.source_ids[j],:]) 
                    WH_sup = np.dot(self.W[:,self.source_ids[j]], self.H_sup[self.source_ids[j],:])
                    loss_full[j,i+1] = loss_func(U[j], U_adv[j], U_sup[j], WH, WH_adv, WH_sup)
        if conv:
            return loss_full

    def fit_std_semi(self, V):
        """
        Semi supervised fitting. Assumes self.W is initialized already, which is handled by NMF_separation
        """

        self.M = V.shape[0]
        self.N = V.shape[1]
        self.S = len(self.ds)

        # Calculate Number of Batches
        if self.batch_size == None:
            self.batch_size = V.shape[1]
        if self.prob == "sup" and self.batch_size_sup is not None:
            self.batch_size = self.batch_size_sup

        self.initH(V = V, prob = "semi")

        N_batches = self.N//self.batch_size 

        # List of ids that will be shuffled
        ids = np.arange(0,self.N)

        # Define updates and loss func, leftover from old code
        W_update = self.std_W_semi_update
        H_update = self.H_semi_update

        for i in range(self.epochs):

            # Shuffle ids, U and H
            np.random.shuffle(ids)

            # Update W for each batch
            for b in range(N_batches):

                batch_ids = ids[np.arange(b*self.batch_size,(b+1)*self.batch_size)%self.N]

                for j in range(self.S):
                
                    # Makes copies of the shuffled data
                    V_batch = V[:,batch_ids]
                    H_batch = self.H[:,batch_ids]

                    # Update W for each batch
                    self.W[:,self.source_ids[-1]] *= W_update(V_batch, self.W, H_batch)

            
            for j in range(self.S):
                # Update H for each source
                self.H[self.source_ids[j],:] *= H_update(V,self.W,self.H, source = j)
            
            if self.normalize:
                norms = np.linalg.norm(self.W, axis = 0)
                self.W = self.W/(norms + 1e-10)
                self.H = norms[:,np.newaxis] * self.H

    # Semi-supervised and adversarial fitting, never used.
    #def fit_adv_semi(self, V, U_adv):
    #    """
    #    Semi supervised fitting. Assumes self.W is initialized already
    #    """

    #    self.M = V.shape[0]
    #    self.N_r = V.shape[1]
    #    self.N_adv = U_adv.shape[1]
    #    self.S = len(self.ds)

    #    if self.batch_size == None:
    #        self.batch_size = self.N_r
    #    if self.batch_size_z == None:
    #        self.batch_size_z = self.N_adv

    #    N_batches = self.N_r//self.batch_size
    #    N_batches_adv = self.N_adv//self.batch_size_z  
    #    
    #    # Used for inner epoch loop
    #    if self.true_sample == "std":
    #        N_batches = N_batches
    #    elif self.true_sample == "adv":
    #        N_batches = N_batches_adv
    #    else:
    #        N_batches = np.minimum(N_batches, N_batches_adv)

    #    self.initH(V = V, prob = "semi_adv")

    #    # List of ids that will be shuffled
    #    ids = np.arange(0,self.N_r)
    #    ids_adv = np.arange(0,self.N_adv)

    #    # Define updates and loss func, leftover from old code
    #    W_update = self.adv_W_semi_update

    #    for i in range(self.epochs):

    #        # Shuffle ids, U and H
    #        np.random.shuffle(ids)
    #        np.random.shuffle(ids_adv)

    #        for b in range(N_batches):

    #            batch_ids = ids[np.arange(b*self.batch_size,(b+1)*self.batch_size)%self.N_r]
    #            batch_ids_adv = ids_adv[np.arange(b*self.batch_size_z,(b+1)*self.batch_size_z)%self.N_adv]

    #            for j in range(self.S):

    #                # Update H for each source
    #                self.H[self.source_ids[j],:] *= self.H_semi_update(V,self.W,self.H, source = j)
    #            
    #            self.H_adv *= self.H_update(U_adv,self.W[:,self.source_ids[-1]],self.H_adv)
    #            
    #            # Makes copies of the shuffled data
    #            V_batch = V[:,batch_ids]
    #            U_adv_batch = U_adv[:,batch_ids_adv]
    #            H_batch = self.H[:,batch_ids]
    #            H_adv_batch = self.H_adv[:,batch_ids_adv]

    #            # Update W for each batch
    #            self.W[:,self.source_ids[-1]] *= W_update(V_batch, U_adv_batch, self.W, H_batch, H_adv_batch)

    #        if self.normalize:
    #            norms = np.linalg.norm(self.W, axis = 0)
    #            self.W = self.W/(norms + 1e-10)
    #            self.H = norms[:,np.newaxis] * self.H
    #            self.H_adv = norms[np.sum(self.ds[:-1]),np.newaxis] * self.H_adv
            
            
                
    def transform(self, U, current = False, WtW = None, WtU = None):
        """
        Solves
        H(U,W) = arg min_{H \ge 0} \|U - WH\|_F^2 + \mu_H |H|_1

        Current uses H that was used in fitting.
        """ 
        if current:
            return self.H
        if WtW is None:
            WtW = np.dot(self.W.T, self.W)
        if WtU is None:
            WtU = np.dot(self.W.T,U)
        N = U.shape[1]
        if WtU is not None:
            d = WtU.shape[0]
        else:
            d = self.d

        H = np.random.uniform(0,1,(d,N))
        
        for i in range(self.test_epochs):

            H *= self.H_update(U = U, H = H, W = self.W, WtU = WtU, WtW = WtW)

        return H

    def transform_source(self,U, source, current = False, WtW = None, WtU = None):
        if current:
            return self.H[self.source_ids[source], :]
        if WtW is None:
            WtW = np.dot(self.W[:,self.source_ids[source]].T, self.W[:,self.source_ids[source]])
        if WtU is None:
            WtU = np.dot(self.W[:,self.source_ids[source]].T,U)
        N = U.shape[1]
        if WtU is not None:
            d = WtU.shape[0]
        else:
            d = self.ds[source]

        H = np.random.uniform(0,1,(d,N))
        
        for i in range(self.test_epochs):
            H *= self.H_update_source(U=U,W=self.W[:,self.source_ids[source]],H=H, source=source, WtU = WtU, WtW = WtW)

        return H

    
    
class NMF_separation:
    """
    A class for source separation using Non-negative Matrix Factorization (NMF).

    Parameters:
    -----------
    ds: list
        A list of integers representing the number of basis vectors in each source.
    NMFs: list, optional
        A list of pre-initialized NMF objects, one for each source. If not provided, new NMF objects will be created.
    Ws: list, optional
        A list of initial W matrices, one for each source. If not provided, W matrices will be initialized randomly.
    betas: np.ndarray, optional
        A 1D array of shape (n_sources) containing the weight of the mixed data in the adversarial dataset.
    omegas: np.ndarray, optional
        A 2D array of shape (n_sources - 1,n_sources - 1) containing the weight of each source data in adversarial dataset.
    wiener: bool, optional
        Whether to apply Wiener filtering to the estimated sources. Default is True.
    prob: str, optional
        Type of NMF method. "std" for standard, "adv" for ANMF, "sup" for DNMF, "full" for D+ANMF.
    eps: float, optional
        A small value added to avoid division by zero errors. Default is 1e-10.
    tau_A: float or list of floats, optional
        Adversarial weight parameter(s). If a single float is provided, it will be used for all sources.
    tau_S: float, optional
        Strong supervision weight parameter.
    warm_start_epochs: int, optional
        The number of epochs to run with a fixed W matrix before updating it. Default is 0.
    **NMF_args:
        Additional arguments to be passed to the NMF object(s).
    """
    def __init__(self, ds, NMFs = None, Ws = None, betas = None, omegas = None, wiener = True, prob = "std", eps = 1e-10,
        tau_A = 0.1, tau_S = 0.5, mu_H = 1e-10, warm_start_epochs = 0, **NMF_args):
  
        self.ds = ds
        self.S = len(ds)
        self.prob = prob
        self.wiener = wiener
        self.betas = betas
        self.omegas = omegas
        self.eps = eps
        self.warm_start_epochs = warm_start_epochs
        
        if isinstance(tau_A,(int,float)):
            self.tau_A = [tau_A]*self.S
        else:
            self.tau_A = tau_A

        if isinstance(mu_H,(int,float)):
            self.mu_H = np.array([mu_H]*np.prod(self.ds))
        elif type(mu_H) == list:
            if len(mu_H) == len(self.ds):
                self.mu_H = np.array([mu_H[i] for i in range(len(self.ds)) for _ in range(self.ds[i])])
        else:
            self.mu_H = np.array(mu_H)
        self.tau_S = tau_S
        
        if NMFs is None:
            self.NMFs = []
            for i,d in enumerate(ds):
                self.NMFs.append(NMF(d, ds = ds, tau_A = self.tau_A[i], mu_H = self.mu_H[i], tau_S = tau_S, prob = prob,
                    W = Ws[i] if Ws is not None else None, **NMF_args))
                    
        else:
            self.NMFs = NMFs
        
        self.NMF_concat = NMF(d = sum(ds), ds = ds, tau_A = self.tau_A[-1], tau_S = tau_S, prob = prob, **NMF_args)
        
        if Ws is not None or NMFs is not None:
            self.to_concat()

    def to_concat(self):
        """
        Concatenates bases in NMFs to NMF_concat 
        """
        Ws = []

        for i in range(self.S):
            Ws.append(self.NMFs[i].W)
        W_concat = np.concatenate(Ws, axis = 1)
        self.NMF_concat.W = W_concat

    def from_concat(self):
        """
        Copies bases from NMF_concat to NMFs 
        """
        for i in range(self.S):
            self.NMFs[i].W = self.NMF_concat.W[:,sum(self.ds[:i]):sum(self.ds[:i+1])]

    def separate(self, V, current = False):
        """
        Separate mixed data given trained bases. 
        """

        if current:
            H_concat = self.NMF_concat.H
        else:  
            H_concat = self.NMF_concat.transform(V)

        # source, pixel, data
        Us = np.zeros((V.shape[0], self.S, V.shape[1]))

        for i in range(self.S):
            H = H_concat[sum(self.ds[:i]):sum(self.ds[:i+1])]
            Us[:,i,:] = np.dot(self.NMFs[i].W, H)
        
        if self.wiener == True:
            U_sum = np.sum(Us, axis = 1) + self.eps
            for i in range(self.S):
                Us[:,i,:] = np.clip(V * Us[:,i,:]/U_sum, a_min = 0.0, a_max = np.max(V))
        return Us
    
    def fit(self, U = None, V = None, U_sup = None, V_sup = None):
        """
        Fits NMF bases for all sources for all different problem settings

        Input:
            U: list of weak supervision datasets for each source
            V: Weak supervision mixed data
            U_sup: list of strong supervision datasets for each source
            V: Strong supervision mixed data so that the i-th data corresponds
                to the i-th data in each dataset in U_sup.
        """

        if self.prob == "std":

            for i,nmf in enumerate(self.NMFs):
                if U_sup is not None and U is not None:
                    nmf.fit_std(np.concatenate((U[i], U_sup[i]), axis = -1))
                else:
                    nmf.fit_std(U[i])
            self.to_concat()
        
        elif self.prob == "adv":

            if U_sup is not None and V_sup is not None and U is not None and V is not None:
                V_ = np.concatenate((V, V_sup), axis = -1)
                U_ = []
                for i in range(self.S):
                    U_.append(np.concatenate((U[i],U_sup[i]), axis = -1))
            else:
                U_ = U
                V_ = V

            U_adv = self.create_adversarial(U_, V = V_)
            for i,nmf in enumerate(self.NMFs):
                nmf.omega = self.omegas[i]
                if self.warm_start_epochs > 0:
                    nmf.prob = "std"
                    nmf.epochs = self.warm_start_epochs
                    nmf.fit_std(U_[i])
                    nmf.epochs = self.epochs
                    nmf.prob = "adv"
                nmf.fit_adv(U_[i], U_adv[i])
            self.to_concat()
        
        elif self.prob == "exem":

            for i,nmf in enumerate(self.NMFs): 
                if U_sup is not None:
                    nmf.fit_exem(np.concatenate((U[i], U_sup[i]), axis = -1))
                else:
                    nmf.fit_exem(U[i])
            self.to_concat()

        elif self.prob == "sup":

            assert U_sup is not None and V_sup is not None, "Discriminative fitting requires U_sup and V_sup, but at least one is None"

            if self.warm_start_epochs > 0:
                for i,nmf in enumerate(self.NMFs):
                    nmf.prob = "std"
                    nmf.epochs = self.warm_start_epochs
                    nmf.fit_std(U[i] if U is not None else U_sup[i])
                    nmf.epochs = self.epochs
                    nmf.prob = "sup"
                self.to_concat()
            self.NMF_concat.fit_sup(U_sup, V_sup)
            self.from_concat()

        elif self.prob == "full":

            assert U is not None and U_sup is not None and V_sup is not None, "Full fitting requires U, U_sup and V_sup"

            U_adv = self.create_adversarial(U, V = V)

            if self.warm_start_epochs > 0:
                for i,nmf in enumerate(self.NMFs):
                    nmf.prob = "std"
                    nmf.epochs = self.warm_start_epochs
                    nmf.fit_std(U[i])
                    nmf.epochs = self.epochs
                    nmf.prob = "full"
                self.to_concat()
            self.NMF_concat.omegas = self.omegas
            self.NMF_concat.fit_full(U, U_adv, U_sup, V_sup)
            self.from_concat()
        
        elif self.prob == "semi":
            for i,nmf in enumerate(self.NMFs):
                if i < len(self.ds) - 1:
                    nmf.prob = "std"
                    nmf.fit_std(U[i])
                else:
                    nmf.M = V.shape[0]
                    nmf.N = V.shape[1]
                    if nmf.W is None:
                        nmf.init = "rand"
                        nmf.initW()
            self.to_concat() 
            self.NMF_concat.fit_std_semi(V = V)
            self.from_concat()
        
        elif self.prob == "semi_adv":
            U_adv = self.create_adversarial(U, V = V)
            for i,nmf in enumerate(self.NMFs):
                if i < len(self.ds) - 1:
                    nmf.prob = "adv"
                    nmf.omega = self.omegas[i]
                    nmf.fit_adv(U[i], U_adv[i])
                else:
                    nmf.M = V.shape[0]
                    nmf.N = V.shape[1]
                    if nmf.W is None:
                        nmf.init = "rand"
                        nmf.initW()
            self.to_concat() 
            #self.NMF_concat.fit_adv_semi(V = V, U_adv = U_adv[-1])
            self.NMF_concat.fit_std_semi(V = V)
            self.from_concat()

        elif self.prob == "semi_exem":
            for i,nmf in enumerate(self.NMFs):
                if i < len(self.ds) - 1:
                    nmf.prob = "exem"
                    nmf.fit_exem(U[i])
                else:
                    nmf.init = "rand"
                    nmf.M = V.shape[0]
                    nmf.N = V.shape[1]
                    nmf.initW()
                    nmf.initH()
            self.to_concat() 
            self.NMF_concat.fit_std_semi(V = V)
            self.from_concat() 
    
    def eval(self, U_test, V_test, metric = 'norm', aggregate = "mean", weights = None, current = False, axis = 0):
        """
        Tests a fitted method on the test data

        TO DO: metric should instead pass a suitable function

        input:
            metric: string, 'norm' for squared Frobenius distrance, 'psnr' for PSNR
            aggregate: string, what aggregation function to use:
                - None, no aggregation
                - 'mean', mean aggregation with np.mean
                - 'median', median aggregation with np.median
                - 'average', weighted average aggregation with np.average. 
                    Defaults to mean if no suitable weights are given.
        """
        out = self.separate(V = V_test, current = current)

        if aggregate == "average" and weights is None:
            weights = [1.0/len(U_test)]*len(U_test)

        if metric == 'norm':
            result = np.linalg.norm(out - U_test, axis = 0)**2
        elif metric == 'psnr':
            result = PSNR(U_test, out)
        
        if aggregate is None:
            return result
        elif aggregate == "mean":
            return np.mean(result, axis = axis)
        elif aggregate == "average":
            return np.average(result, weights = weights, axis = axis)
        elif aggregate == "median":
            return np.median(result, axis = axis)

    def create_adversarial(self, U, V = None):
        """
        Generates adversarial dataset that potentially contains both mixed data and data from other sources
        U_{Z_i} = tau_i * [sqrt(omega_i1 N_adv/N_1)U_1 ... sqrt((1- sum_{j \neq i} omega_{ij}) N_adv/N_mix ) beta_i V]

        Input:
            U: List of S (M,N) numpy arrays containing true data of each source
            V: (M,N) numpy array containing mixed data
            taus: List of S floats. Controls the overall weight of the adversarial term. Defaults to $1$
            betas: List of S floats. Controlls the weight of the mixed data. Defaults to 1.
            omegas: 2-D list of (S-1)**2 floats. Controls the weights of the individual sources. Defaults to N_i/\hat{N}_i
        Output:
            U_adv: List of S arrays each of size (M,N_{Z_i})
        """
        U_adv = []
        Ns = []
        useMixed = (V is not None)
        S = len(U)
        for i in range(S):
            Ns.append(U[i].shape[1])

        N_advs = np.zeros(S, dtype = np.int32)
        # All my homies hate list comprehension
        for i in range(S):
            N_advs[i] += np.sum(Ns[:i] + Ns[i+1:])
        for i in range(S):
            if useMixed:
                N_advs[i] += V.shape[1]
        if self.betas is None:
            self.betas = [1.0] * (S + int(useMixed))
        if self.omegas is None:
            self.omegas = []
            for i in range(S):
                self.omegas.append(np.zeros(N_advs[i]))
            #self.omegas = np.zeros((S, S-1))

            # For each source we need to create a dataset 
            for i in range(S):
                # We need to account for the weight of the reminaing S-1 socources after taking one source out
                curr = 0
                for j in range(S-1):
                    #self.omegas[i,j] = Ns[j + (j >= i)]/N_advs[i]
                    self.omegas[i][curr:curr + Ns[j + (j>=i)]] = Ns[j + (j>=i)]/N_advs[i]
                    curr += Ns[j + (j>=i)]
                if useMixed:
                    self.omegas[i][curr:] = V.shape[1] * self.betas[i]/N_advs[i]

        for i in range(len(U)):
            U_ = U[:i] + U[i+1:]
            for j in range(len(U_)):
                #U_[j] = np.sqrt(self.omegas[i,j] * N_advs[i]/Ns[j + (j >= i)]) * U_[j]
                U_[j] = np.sqrt(Ns[j + (j>=i)]/N_advs[i]) * U_[j]
            if useMixed:
                #U_.append(np.sqrt((1.0 - np.sum(self.omegas[i])) * N_advs[i]/V.shape[1]) * self.betas[i] * V)
                U_.append(np.sqrt((1.0 - (sum(Ns) - Ns[i])/N_advs[i])) * self.betas[i] * V)

            U_adv.append(np.copy(np.concatenate(U_, axis = 1)))

            #if self.prob == "adv" or self.prob == "semi_adv":
                #U_adv.append(np.copy(np.sqrt(self.tau_A[i]) * np.concatenate(U_, axis = 1)))
            #    U_adv.append(np.copy(np.concatenate(U_, axis = 1)))

            #elif self.prob == "full":
                #U_adv.append(np.copy(np.sqrt((1.0 - self.tau_S) * self.tau_A[i]) * np.concatenate(U_, axis = 1)))
            #    U_adv.append(np.copy(np.concatenate(U_, axis = 1)))

        return U_adv

class random_search:
    """
    Class for random search with CV.
    """
    def __init__(self,method,param_dicts, N_ex= 50, metric = "psnr", source_aggregate = "mean", data_aggregate = "mean", cv = 0, verbose = False):
        """
        Input:
            method, class with fit, separate and eval functions, like NMF_separation
            param_dicts, list of dictionaries. Each dictionary contains:
                'name': string, which is what will be passed to the __init__ function of method.
                'dist': function, returns a candidate we want to search.
                    Distribution should also handle if parameter is discrete or continuous
                    ex1: lambda : np.random.uniform(0.0,1.0)
                    ex2: lambda : np.random.randint(0,10)
                    ex3: lambda : np.random.choice([10,25,50], replace = True)
                    ex4: lambda : [0.5,0.5]
            cv: int, number of cv iterations for supervised data
            metric: string, metric to be used, "norm" or "psnr".
            source_aggregate: string, aggregation for sources, see eval in NMF_separation
            data_aggregate: string, aggregation of data after source aggregation. "mean" or "median".

        """
        self.method = method
        self.param_dicts = param_dicts
        self.N_ex = N_ex
        self.metric = metric
        self.cv = cv
        self.source_aggregate = source_aggregate
        self.data_aggregate = data_aggregate
        self.verbose = verbose

        self.best_model = None
        self.best_param = None
        self.best_val = - np.inf

    def fit(self, U = None, V = None, U_sup = None, V_sup = None, refit = False):
        """
        Fits
        """
        if self.cv > 1:
            # Get the number of samples along the second axis
            n = V_sup.shape[1]
    
            # Calculate the size of each split
            split_size = n // self.cv
        else:
            U_s = np.zeros((U_sup[0].shape[0], len(U_sup), U_sup[0].shape[1]))
            for i in range(len(U_sup)):
                U_s[:,i,:] = U_sup[i]

        # Results should be stored here!
        results = {}
        for param in self.param_dicts:
            results[param["name"]] = []
        results["val"] = []

        for i in range(self.N_ex):

            # Get params for this experiment
            params = {}
            for j,param in enumerate(self.param_dicts):
                    params[param["name"]] = param["dist"]()
            
            if self.cv > 1:
                # Value to store metric
                ex_val = 0.0

                # Iterate over each cv fold 
                for j in range(self.cv):
                    # Get the start and end indices of the test split
                    start = j * split_size
                    end = (j + 1) * split_size
        
                    # Get the indices for the train data
                    train_idx = np.arange(n)
                    train_idx = np.delete(train_idx, np.arange(start, end))
        
                    # Get the train and test data
                    train_U = []
                    test_U = np.zeros((U_sup[0].shape[0], len(U_sup), end-start))
                    for k in range(len(U_sup)):
                        train_U.append(U_sup[k][:,train_idx])
                        test_U[:,k,:] = U_sup[k][:,start:end]    
                    train_mix = V_sup[:, train_idx]
                    test_V = V_sup[:, start:end]

                    sep = self.method(**params)

                    sep.fit(U = U, V = V, U_sup = train_U, V_sup = train_mix)
                    dists = sep.eval(test_U, test_V, metric = self.metric, aggregate = self.source_aggregate)
                    if self.data_aggregate == "mean":
                        ex_val += np.mean(dists) / self.cv
                    elif self.data_aggregate == "median":
                        ex_val += np.median(dists) / self.cv
            else:
                sep = self.method(**params)
                sep.fit(U = U, V = V, U_sup = U_sup, V_sup = V_sup)

                # Evaluates 
                dists = sep.eval(U_s, V_sup, metric = self.metric, aggregate = self.source_aggregate)

                if self.data_aggregate == "mean":
                    ex_val = np.mean(dists)
                elif self.data_aggregate == "median":
                    ex_val = np.median(dists)
            
            for key in params:
                results[key].append(params[key])
            results["val"].append(ex_val)

            if ex_val > self.best_val:
                self.best_val = ex_val
                self.best_param = params
                if not refit:
                    self.best_model = deepcopy(sep)
        if refit:
            self.best_model = self.method(**self.best_param)
            self.best_model.fit(U = U, V = V, U_sup = U_sup, V_sup = V_sup)

        if self.verbose:
            print("Best param", self.best_param, self.best_val)
        return results
