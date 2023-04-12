import numpy as np
from utils import *
from copy import deepcopy
from librosa import stft, istft
from museval.metrics import bss_eval

class NMF:

    def __init__(self, d = None, ds = None, tau_A = 0.1, tau_S = 0.0,
        W = None, prob = "std", init = "exem", update_H = False,
        epochs = 50, test_epochs = 50, warm_start_epochs = 0,
        batch_size = None, batch_size_z = None, batch_size_sup = None,
        true_sample = "std", normalize = False,
        mu_W = 1e-6, mu_H = 1e-6):
        """
        Class for Non-Negative Matrix Factorization for source separation. 
        Decomposes Non-negative data stored columnwise in and m times n array U as U approx WH,
        where W is m times d and H is d times n.

        Class has several fit functions for different variants of NMF.
        
        Parameters:
        -----------
        mu_W (float): regularization parameter for W
        mu_H (float): regularization parameter for H
        prob (str): type of NMF. "std" for standard, "adv" for ANMF, 
                    "sup" for DNMF and "full" for D+ANMF.
        init (str): initialization method. "random" for random initialization and 
                    "exem" for exemplar-based initialization.
        epochs (int): number of training epochs.
        test_epochs (int): number of epochs used in testing.
        warm_start_epochs (int): number of warm-start epochs before the main training.
        d (int): number of basis vectors in W.
        ds (list): a list of the number of basis vectors for each source where it is needed
        batch_size (int): size of mini-batches for true data term.
        batch_size_z (int): size of mini-batches for adversarial data term
        batch_size_sup (int): size of mini-batches for strong supervision data term.
        true_sample (str): which dataset to fully sample. Data is shuffled and new epoch is started
                            when all data in this dataset has been passed through. 
                           "std" for true data, "adv" for adversarial data, 
                           "sup" for supervised data.
        tau_A (float): Adversarial weight for ANMF and D+ANMF.
        tau_S (float): Strong supervision weight for D+ANMF.
        normalize (bool): whether or not to normalize the columns of W.
        update_H (bool): whether or not to do batchwise updates of H during training. 
        """
        self.mu_W = mu_W
        self.mu_H = mu_H
        self.prob = prob
        self.init = init
        self.epochs = epochs
        self.test_epochs = test_epochs
        self.warm_start_epochs = warm_start_epochs
        self.d = d
        self.ds = ds
        self.batch_size = batch_size
        self.batch_size_z = batch_size_z
        self.batch_size_sup = batch_size_sup
        self.true_sample = true_sample
        self.tau_A = tau_A
        self.tau_S = tau_S
        self.normalize = normalize
        self.update_H = update_H

        # Copy W if it is given
        self.W = W

        # Initialize H_r, H_z and H_sup
        self.H_r = None
        self.H_z = None
        self.H_sup = None

        # Lengths
        self.M = None
        self.N_r = None
        self.N_z = None
        self.N_sup = None

        # Number of sources
        self.S = None

        # Calculate what parts of W and H_sup correspond
        # to what class in relevant cases
        if self.ds is not None:
            self.class_ids = []
            for i in range(len(self.ds)):
                self.class_ids.append(np.arange(sum(self.ds[:i]), sum(self.ds[:i+1])))

    def std_W_update(self,U_r,W,H_r, H_rH_rT = None):
        """
        Update rule for W for standard NMF and discriminative NMF
        """
        if H_rH_rT is None:
            H_rH_rT = np.dot(H_r, H_r.T)
        invN_r = 1.0/U_r.shape[1]

        return np.dot(U_r, H_r.T) * invN_r / (np.dot(W, H_rH_rT) * invN_r + self.mu_W)

    def adv_W_update(self, U_r, U_z, W, H_r, H_z, U_rH_rT = None, U_zH_zT = None, H_rH_rT = None, H_zH_zT = None):
        """
        Update rule for adversarial NMF
        """
        if U_rH_rT is None:
            U_rH_rT = np.dot(U_r,H_r.T)
        if U_zH_zT is None:
            U_zH_zT = np.dot(U_z,H_z.T)
        if H_rH_rT is None:
            H_rH_rT = np.dot(H_r,H_r.T)
        if H_zH_zT is None:
            H_zH_zT = np.dot(H_z,H_z.T)

        invN_r = 1.0/U_r.shape[1]
        invN_z = 1.0/U_z.shape[1]

        return (np.dot(W,H_zH_zT) * invN_z + U_rH_rT * invN_r)/(np.dot(W, H_rH_rT)*invN_r + U_zH_zT * invN_z + self.mu_W)

    def std_W_semi_update(self,V,W,H, HH_rT = None):
        """
        Update rule for W in semi-supervised data setting where data from the last source is missing
        """
        invN_v = 1.0/V.shape[1]
        if HH_rT is None:
            HH_rT = np.dot(H, H[self.class_ids[-1],:].T)
        return np.dot(V,H[self.class_ids[-1],:].T) * invN_v/(np.dot(W, HH_rT) * invN_v + self.mu_W)

    # Semi supervised + adversarial update, never used
    #def adv_W_semi_update(self,V,U_z, W, H, H_z, VH_rT = None, U_zH_zT = None, HH_rT = None, HH_zT = None):
    #    """
    #    Update rule for W for standard NMF in semi-supervised data setting 
    #    """
    #    invN_v = 1.0/V.shape[1]
    #    invN_z = 1.0/U_z.shape[1]
    #    if VH_rT is None:
    #        VH_rT = np.dot(V, H[self.class_ids[-1],:].T)
    #    if U_zH_zT is None:
    #        U_zH_zT = np.dot(U_z, H_z.T)
    #    if HH_rT is None:
    #        HH_rT = np.dot(H, H[self.class_ids[-1],:].T)
    #    if HH_zT is None:
    #        HH_zT = np.dot(H_z, H_z.T)

    #    return (np.dot(V,H[self.class_ids[-1],:].T) * invN_v + np.dot(W[:,self.class_ids[-1]],HH_zT) * invN_z)/(np.dot(W, HH_rT) * invN_v + U_zH_zT * invN_z + self.mu_W)

    def H_update(self,U,W,H, WtU = None, WtW = None):
        """
        Update rule for H for standard NMF and discriminative NMF
        """
        if WtU is None:
            WtU = np.dot(W.T, U)
        if WtW is None:
            WtW = np.dot(W.T, W)

        #return WtU/(np.dot(WtW, H) + self.mu_H)

        invN = 1.0/U.shape[1]

        return (WtU * invN)/(np.dot(WtW, H) * invN + self.mu_H)

    def H_semi_update(self,V,W,H,source,WitW = None):
        """
        Update rule for H in semi-supervised setting where data from the last source is missing
        """
        
        if WitW is None:
            WitW = np.dot(W[:,self.class_ids[source]].T, W)
        
        invN_V = 1.0/V.shape[1]

        return (np.dot(W[:,self.class_ids[source]].T, V) * invN_V)/(np.dot(WitW, H) * invN_V + self.mu_H)


    def initializeWH(self, U_r = None, U_z = None, U_sup = None, V_sup = None, prob = "std"):
        """
        Initialize W and H for fitting

        TO DO: Move this into initW and initH
        """

        if self.ds is not None:
            class_ids = []
            for i in range(len(self.ds)):
                class_ids.append(np.arange(sum(self.ds[:i]), sum(self.ds[:i+1])))

        if self.W is None:
            if self.init == "rand":
                self.W = np.random.uniform(0,1,(self.M,self.d))
            elif self.init == "exem":
                if prob == "std" or prob == "adv" or prob == "exem":
                    self.W = U_r[:,np.random.choice(self.N_r, size = self.d, replace = False)]
                else:
                    self.W = np.zeros((self.M,self.d))
                    for j in range(len(self.ds)):
                        if U_r is not None:
                            self.W[:,class_ids[j]] = U_r[j][:,np.random.choice(self.N_r, size = self.ds[j], replace = False)]
                        else:
                            self.W[:,class_ids[j]] = U_sup[j][:,np.random.choice(self.N_sup, size = self.ds[j], replace = False)]

        if self.normalize == True:
            norms = np.linalg.norm(self.W, axis = 0)
            self.W = self.W/(norms + 1e-10)

        if self.init == "rand":
            if prob != "sup":
                self.H_r = np.random.uniform(0,1,(self.d,self.N_r))
                if prob == "adv" or prob == "full":
                    self.H_z = np.random.uniform(0,1,(self.d,self.N_z))
            if prob == "sup" or prob == "full":
                self.H_sup = np.random.uniform(0,1,(self.d,self.N_sup))

        elif self.init == "exem":
            if prob == "std" or prob == "adv" or prob == "exem":
                self.H_r = self.transform(U_r)
                if prob == "adv":
                    self.H_z = self.transform(U_z)
            elif prob == "full":
                self.H_r = np.zeros((self.d, self.N_r))
                self.H_z = np.zeros((self.d, self.N_z))
                for j in range(len(self.ds)):
                    WtW = np.dot(self.W[:,class_ids[j]].T, self.W[:,class_ids[j]])
                    self.H_r[class_ids[j],:] = self.transform(U_r[j], WtW = WtW, WtU = np.dot(self.W[:,class_ids[j]].T, U_r[j])) 
                    self.H_z[class_ids[j],:] = self.transform(U_z[j], WtW = WtW, WtU = np.dot(self.W[:,class_ids[j]].T, U_z[j])) 
            if prob == "sup" or prob == "full":
                self.H_sup = self.transform(V_sup)
    
    def initW(self,U_r = None, U_z = None, U_sup = None, V_sup = None, prob = "std"):
        """
        Initialize W 

        TO DO: Remove prob argument, use class attribute
        """
        if self.W is None:
            if self.init == "rand":
                self.W = np.random.uniform(0,1,(self.M,self.d))
            elif self.init == "exem":
                if prob == "std" or prob == "adv" or prob == "exem":
                    self.W = U_r[:,np.random.choice(self.N_r, size = self.d, replace = False)]
                else:
                    self.W = np.zeros((self.M,self.d))
                    for j in range(len(self.ds)):
                        if U_r is not None:
                            self.W[:,self.class_ids[j]] = U_r[j][:,np.random.choice(self.N_r, size = self.ds[j], replace = False)]
                        else:
                            self.W[:,self.class_ids[j]] = U_sup[j][:,np.random.choice(self.N_sup, size = self.ds[j], replace = False)]

        if self.normalize == True:
            norms = np.linalg.norm(self.W, axis = 0)
            self.W = self.W/(norms + 1e-10)
    
    def initH(self,U_r = None, U_z = None, V = None, U_sup = None, V_sup = None, prob = "std"):
        """
        Initialize latent variables 

        TO DO: Remove prob argument, use class attribute
        """

        if self.init == "rand":
            if prob != "sup":
                self.H_r = np.random.uniform(0,1,(self.d,self.N_r))
                if prob == "adv" or prob == "full" or prob == "semi_adv":
                    self.H_z = np.random.uniform(0,1,(self.d,self.N_z))
            if prob == "sup" or prob == "full":
                self.H_sup = np.random.uniform(0,1,(self.d,self.N_sup))

        elif self.init == "exem":
            assert self.W is not None, "Exemplar based initialization requires W not None"
            if prob == "std" or prob == "adv" or prob == "exem":
                self.H_r = self.transform(U_r)
                if prob == "adv":
                    self.H_z = self.transform(U_z)
            elif prob == "full":
                self.H_r = np.zeros((self.d, self.N_r))
                self.H_z = np.zeros((self.d, self.N_z))
                for j in range(len(self.ds)):
                    WtW = np.dot(self.W[:,self.class_ids[j]].T, self.W[:,self.class_ids[j]])
                    self.H_r[self.class_ids[j],:] = self.transform(U_r[j], WtW = WtW, WtU = np.dot(self.W[:,self.class_ids[j]].T, U_r[j])) 
                    self.H_z[self.class_ids[j],:] = self.transform(U_z[j], WtW = WtW, WtU = np.dot(self.W[:,self.class_ids[j]].T, U_z[j])) 
            elif prob == "semi" or prob == "semi_adv":
                self.H_r = self.transform(V)
                if prob == "semi_adv":
                    self.H_z = np.random.uniform(0,1,(self.ds[-1],self.N_z))
            if prob == "sup" or prob == "full":
                self.H_sup = self.transform(V_sup)

    def std_loss(self,U_r, WH_r):
        """
        Loss for standard NMF
        """
        return 1/U_r.shape[1] * np.linalg.norm(U_r - WH_r, 'fro')**2
    
    def adv_loss(self,U_r,U_z,WH_r,WH_z):
        """
        Loss for ANMF

        TO DO: Double check if tau_A is multiplied redundantly here, most likely it can be removed.
        """
        return 1/U_r.shape[1] * np.linalg.norm(U_r - WH_r, 'fro')**2 - self.tau_A/U_z.shape[1] * np.linalg.norm(U_z - WH_z, 'fro')**2

    def fit_std(self, U_r, conv = False):
        """
        Fits standard NMF by solving

        min_{W \ge 0} 1/N \|U - WH(U,W)\|_F^2
        where H(U,W) = arg min_{H \ge 0} \|U - WH\|_F^2
        """

        self.M = U_r.shape[0]
        self.N_r = U_r.shape[1]

        # Calculate Number of Batches
        if self.batch_size == None:
            self.batch_size = U_r.shape[1]
        if self.prob == "sup" and self.batch_size_sup is not None:
            self.batch_size = self.batch_size_sup

        N_batches = self.N_r//self.batch_size 

        # Initialize if nothing exists
        if self.W is None or self.H_r is None:
            self.initializeWH(U_r = U_r, prob = "std")

        # List of ids that will be shuffled
        ids = np.arange(0,self.N_r)


        # Define updates and loss func, leftover from old code
        W_update = self.std_W_update
        H_update = self.H_update

        # Define array we will need for convergence
        if conv:
            loss_func = self.std_loss
            loss_std = np.zeros((self.epochs + 1))
            WH_r = np.dot(self.W,self.H_r)
            loss_std[0] = loss_func(U_r,WH_r)

        for i in range(self.epochs):
            # Shuffle ids, U and H
            np.random.shuffle(ids)

            # Update H
            if not self.update_H:
                self.H_r *= H_update(U_r,self.W,self.H_r)

            for b in range(N_batches):
                
                # Technically do not need the np.arange here
                batch_ids = ids[np.arange(b*self.batch_size,(b+1)*self.batch_size)%self.N_r]

                # Makes copies of the shuffled data
                U_r_batch = U_r[:,batch_ids]
                H_r_batch = self.H_r[:,batch_ids]

                if self.update_H:

                    H_r_batch *= H_update(U_r_batch,self.W,H_r_batch)

                    self.H_r[:,batch_ids] = H_r_batch

                # Update W for each batch
                self.W *= W_update(U_r_batch, self.W, H_r_batch)

            if self.normalize:
                norms = np.linalg.norm(self.W, axis = 0)
                self.W = self.W/(norms + 1e-10)
                self.H_r = norms[:,np.newaxis] * self.H_r


            if conv:
                WH_r = np.dot(self.W,self.H_r)
                loss_std[i+1] = loss_func(U_r, WH_r)
        if conv:
            return loss_std

    def fit_adv(self, U_r, U_z, conv = False):
        """
        Fits adversarial NMF by solving

        min_{W \ge 0} 1/N \|U - WH(U,W)\|_F^2 - tau_A/\hat{N} \|U - WH(\hat{U},W)\|_F^2 
        where H(U,W) = arg min_{H \ge 0} \|U - WH\|_F^2

        Here U is true data and \hat{U} is adversarial data.
        """
        self.M = U_r.shape[0]
        self.N_r = U_r.shape[1]

        assert self.M == U_z.shape[0], f"U_r has first axis {self.M} and U_z has first axis {U_z.shape[0]} which does not match."

        self.N_z = U_z.shape[1]

        # Calculate umber of Batches
        if self.batch_size == None:
            self.batch_size = self.N_r
        if self.batch_size_z == None:
            self.batch_size_z = self.N_z

        N_batches_r = self.N_r//self.batch_size
        N_batches_z = self.N_z//self.batch_size_z  
        
        # Used for inner epoch loop
        if self.true_sample == "std":
            N_batches = N_batches_r
        elif self.true_sample == "adv":
            N_batches = N_batches_z
        else:
            N_batches = np.minimum(N_batches_r, N_batches_z)

        if self.W is None or self.H_r is None or self.H_z is None:
            self.initializeWH(U_r = U_r, U_z = U_z, prob = "adv")

        # List of ids that will be shuffled
        ids_r = np.arange(0,self.N_r)
        ids_z = np.arange(0,self.N_z)

        # Set update
        W_update = self.adv_W_update
        H_update = self.H_update
        
        if conv:
            loss_func = self.adv_loss
            loss_adv = np.zeros((self.epochs + 1))
            WH_r = np.dot(self.W,self.H_r)
            WH_z = np.dot(self.W,self.H_z)
            loss_adv[0] = loss_func(U_r,U_z, WH_r, WH_z)
            

        for i in range(self.epochs):
            # Shuffle ids, U and H
            np.random.shuffle(ids_r)
            np.random.shuffle(ids_z)


            if not self.update_H: 
                WtW = np.dot(self.W.T, self.W)
                self.H_r *= H_update(U_r,self.W,self.H_r, WtW = WtW)
                self.H_z *= H_update(U_z,self.W,self.H_z, WtW = WtW)

            for b in range(N_batches):

                batch_ids_r = ids_r[np.arange(b*self.batch_size,(b+1)*self.batch_size)%self.N_r]
                batch_ids_z = ids_z[np.arange(b*self.batch_size_z,(b+1)*self.batch_size_z)%self.N_z]

                # Makes copies of the shuffled data
                U_r_batch = U_r[:,batch_ids_r]
                H_r_batch = self.H_r[:,batch_ids_r]
                U_z_batch = U_z[:,batch_ids_z]
                H_z_batch = self.H_z[:,batch_ids_z]

                # Update H for each batch
                if self.update_H:

                    WtW = np.dot(self.W.T, self.W)
                    H_r_batch *= H_update(U_r_batch,self.W,H_r_batch, WtW = WtW)
                    H_z_batch *= H_update(U_z_batch,self.W,H_z_batch, WtW = WtW)

                    self.H_r[:,batch_ids_r] = H_r_batch
                    self.H_z[:,batch_ids_z] = H_z_batch

                # Update W for each batch
                self.W *= W_update(U_r_batch, U_z_batch, self.W, H_r_batch, H_z_batch)

            if self.normalize:
                norms = np.linalg.norm(self.W, axis = 0)
                self.W = self.W/(norms + 1e-10)

                # Assumes true_sample is std !!!!!
                self.H_r *= norms[:,np.newaxis]

                if self.update_H:
                    self.H_z[:,ids_z[:np.minimum(self.N_z, N_batches * self.batch_size_z)]] *= norms[:,np.newaxis]
                else:
                    self.H_z *= norms[:,np.newaxis]

            if conv:
                WH_r = np.dot(self.W,self.H_r)
                WH_z = np.dot(self.W,self.H_z)
                loss_adv[i+1] = loss_func(U_r,U_z, WH_r, WH_z)
                
        if conv:
            return loss_adv

    def fit_sup(self,U_sup,V_sup,conv = False):
        """
        Solves:

        \min_{W \ge 0} \sum_i^{S} \|U_i - W_i H_i\|_F^2
        s.t H = \argmin_{H} \|V - WH\|_F^2,
        where W = [W_1,...W_S] and H = [H_1,... W_S] are concatenated versions of W and H.
        Technically solves this on multi-objective form.

        U_r should be input as a list of (M,N) arrays

        TO DO: Make it handle (M,S,N) arrays as well
        """
        assert self.ds is not None and self.d is not None, "Supervised fitting needs both d and ds"
        assert len(U_sup) == len(self.ds), "U_sup and self.ds shapes do not match"
        #self.Ms = np.zeros(len(U_r), dtype = int)


        self.N_sup = V_sup.shape[1] 
        self.M = V_sup.shape[0]
        self.S = len(U_sup)

        
        # Calculate Number of Batches
        if self.batch_size_sup is None:
            self.batch_size_sup = self.N_sup
        self.N_batches = self.N_sup//self.batch_size_sup 


        if self.W is None or self.H_r is None:
            self.initializeWH(U_sup = U_sup, V_sup = V_sup, prob = "sup")

        ids = np.arange(0,self.N_sup)

        W_update = self.std_W_update
        H_update = self.H_update

        class_ids = []
        for i in range(len(self.ds)):
            class_ids.append(np.arange(sum(self.ds[:i]), sum(self.ds[:i+1])))

        if conv:
            loss_func = self.std_loss
            loss_sup = np.zeros((self.S, self.epochs + 1))
            for j in range(self.S):
                WH_sup = np.dot(self.W[:,class_ids[j]], self.H_sup[class_ids[j],:])
                loss_sup[j,0] = loss_func(U_sup[j], WH_sup)
        
        for i in range(self.epochs):

            # Shuffle data
            np.random.shuffle(ids)

            if not self.update_H:
                self.H_sup *= H_update(V_sup,self.W,self.H_sup) 

            # Iterate over each source

            for b in range(self.N_batches):

                batch_ids = ids[np.arange(b*self.batch_size_sup,(b+1)*self.batch_size_sup)%self.N_sup]

                for j in range(self.S):
                    
                    U_sup_batch = U_sup[j][:,batch_ids]
                    H_sup_batch = self.H_sup[class_ids[j]][:,batch_ids]
                    if self.update_H:
                        V_sup_batch = V_sup[:,batch_ids]
                        self.H_sup[:,batch_ids] *= H_update(V_sup_batch, self.W, self.H_sup[:,batch_ids])

                    # Update W
                    self.W[:,class_ids[j]] *= W_update(U_sup_batch, self.W[:,class_ids[j]], H_sup_batch)
                
            if self.normalize:
                norms = np.linalg.norm(self.W, axis = 0)
                self.W = self.W/(norms + 1e-10)
                self.H_sup = norms[:,np.newaxis] * self.H_sup
            
            if conv:
                for j in range(self.S):
                    WH_sup = np.dot(self.W[:,class_ids[j]], self.H_sup[class_ids[j],:])
                    loss_sup[j,i+1] = loss_func(U_sup[j], WH_sup)
        if conv:
            return loss_sup

    def fit_exem(self, U_r = None):
        """
        Fits Exemplar-based NMF. 
        """

        self.N_r = U_r.shape[1]

        # Fitting of exemplar-based NMF is handled by the initialization function
        self.initW(U_r = U_r)

    
    def fit_full(self, U_r, U_z, U_sup, V_sup, conv = False):
        """
        Fit function that can handle fitting for fitting D+ANMF which includes weak supervision data, adversarial data
        and strong supervision data. 

        TO DO: Implementation that handles unbalanced data. Rewrite so it does not make redundant copies of data.

        """

        # Store sizes
        self.N_r = U_r[0].shape[1]
        self.N_z = U_z[0].shape[1]
        self.N_sup = V_sup.shape[1]
        self.M = V_sup.shape[0]
        self.S = len(U_sup)

        # Initialize W and the different latent variables
        self.initializeWH(U_r = U_r, U_z = U_z, U_sup = U_sup, V_sup = V_sup, prob = "full")

        if self.batch_size == None:
            self.batch_size = self.N_r
        if self.batch_size_z == None:
            self.batch_size_z = self.N_z
        if self.batch_size_sup == None:
            self.batch_size_sup = self.N_sup

        # Calculate number of batches
        N_batches_r = self.N_r//self.batch_size
        N_batches_z = self.N_z//self.batch_size_z 
        N_batches_sup = self.N_sup//self.batch_size_sup

        inv_N_r = 1.0/self.N_r
        inv_N_z = 1.0/self.N_z
        inv_N_sup = 1.0/self.N_sup
        
        # Select which data to true sample
        if self.true_sample == "std":
            N_batches = N_batches_r
        elif self.true_sample == "adv":
            N_batches = N_batches_z
        elif self.true_sample == "sup":
            N_batches = N_batches_sup
        else:
            N_batches = np.minimum(N_batches_r, N_batches_z, N_batches_sup)
        
        class_ids = []
        for i in range(len(self.ds)):
            class_ids.append(np.arange(sum(self.ds[:i]), sum(self.ds[:i+1])))
        

        ids_r = np.arange(0,self.N_r)
        ids_z = np.arange(0,self.N_z)
        ids_sup = np.arange(0,self.N_sup)

        if conv:
            loss_func = lambda U_r, U_z, U_sup, WH_r, WH_z, WH_sup : self.adv_loss(np.sqrt((1 - self.tau_S))* U_r,U_z, WH_r, WH_z) + self.std_loss(np.sqrt(self.tau_S)* U_sup, WH_sup)
            loss_full = np.zeros((self.S, self.epochs + 1))
            for j in range(self.S):
                WH_r = np.dot(self.W[:,class_ids[j]], self.H_r[class_ids[j],:]) 
                WH_z = np.dot(self.W[:,class_ids[j]], self.H_z[class_ids[j],:]) 
                WH_sup = np.dot(self.W[:,class_ids[j]], self.H_sup[class_ids[j],:])
                loss_full[j,0] = loss_func(U_r[j], U_z[j], U_sup[j], WH_r, WH_z, WH_sup)


        # Start main loop
        for i in range(self.epochs):
            # Shuffle data

            np.random.shuffle(ids_r)
            np.random.shuffle(ids_z)
            np.random.shuffle(ids_sup)

            # Update H_sup
            self.H_sup *= self.H_update(np.sqrt(self.tau_S) * V_sup,self.W,self.H_sup) 
            
            # Update W for each source
            for j in range(self.S):

                # Update H_r and H_z for each class
                WtW = np.dot(self.W[:,class_ids[j]].T, self.W[:,class_ids[j]])
                self.H_r[class_ids[j],:] *= self.H_update(np.sqrt(1.0 - self.tau_S) * U_r[j],self.W[:,class_ids[j]],self.H_r[class_ids[j],:], WtW = WtW) 
                self.H_z[class_ids[j],:] *= self.H_update(U_z[j], self.W[:,class_ids[j]],self.H_z[class_ids[j],:], WtW = WtW) 

                # Split data into batches
                U_r_b = np.array_split(np.sqrt(1 - self.tau_S) * U_r[j][:,ids_r], N_batches_r, axis = 1)
                H_r_b = np.array_split(self.H_r[class_ids[j]][:,ids_r], N_batches_r, axis = 1)
                U_z_b = np.array_split(U_z[j][:,ids_z], N_batches_z, axis = 1)
                H_z_b = np.array_split(self.H_z[class_ids[j]][:,ids_z], N_batches_z, axis = 1) 
                U_sup_b = np.array_split(np.sqrt(self.tau_S)*U_sup[j][:,ids_sup], N_batches_sup, axis = 1)
                H_sup_b = np.array_split(self.H_sup[class_ids[j]][:,ids_sup], N_batches_sup, axis = 1)
                
                for b in range(N_batches):
                    # Calculate terms needed for W update
                    top = np.dot(U_r_b[b%N_batches_r], H_r_b[b%N_batches_r].T) * inv_N_r
                    top += np.dot(self.W[:,class_ids[j]], np.dot(H_z_b[b%N_batches_z], H_z_b[b%N_batches_z].T)) * inv_N_z
                    top += np.dot(U_sup_b[b%N_batches_sup], H_sup_b[b%N_batches_sup].T) * inv_N_sup

                    bot = np.dot(self.W[:,class_ids[j]], np.dot(H_r_b[b%N_batches_r], H_r_b[b%N_batches_r].T)) * inv_N_r
                    bot += np.dot(U_z_b[b%N_batches_z], H_z_b[b%N_batches_z].T) * inv_N_z
                    bot += np.dot(self.W[:,class_ids[j]], np.dot(H_sup_b[b%N_batches_sup], H_sup_b[b%N_batches_sup].T)) * inv_N_sup

                    # Calculate W update
                    self.W[:,class_ids[j]] *= (top)/(bot + self.mu_W)
                


            if self.normalize:
                norms = np.linalg.norm(self.W, axis = 0)
                self.W = self.W/(norms + 1e-10)
                self.H_r = norms[:,np.newaxis] * self.H_r
                self.H_z = norms[:,np.newaxis] * self.H_z
                self.H_sup = norms[:,np.newaxis] * self.H_sup

            if conv:
                for j in range(self.S):
                    WH_r = np.dot(self.W[:,class_ids[j]], self.H_r[class_ids[j],:]) 
                    WH_z = np.dot(self.W[:,class_ids[j]], self.H_z[class_ids[j],:]) 
                    WH_sup = np.dot(self.W[:,class_ids[j]], self.H_sup[class_ids[j],:])
                    loss_full[j,i+1] = loss_func(np.sqrt((1.0 - self.tau_S))* U_r[j], U_z[j], np.sqrt(self.tau_S)*U_sup[j], WH_r, WH_z, WH_sup)
        if conv:
            return loss_full

    def fit_std_semi(self, V):
        """
        Semi supervised fitting. Assumes self.W is initialized already, which is handled by NMF_separation
        """

        self.M = V.shape[0]
        self.N_r = V.shape[1]
        self.S = len(self.ds)

        # Calculate Number of Batches
        if self.batch_size == None:
            self.batch_size = V.shape[1]
        if self.prob == "sup" and self.batch_size_sup is not None:
            self.batch_size = self.batch_size_sup

        self.initH(V = V, prob = "semi")

        N_batches = self.N_r//self.batch_size 

        # List of ids that will be shuffled
        ids = np.arange(0,self.N_r)

        # Define updates and loss func, leftover from old code
        W_update = self.std_W_semi_update
        H_update = self.H_semi_update

        for i in range(self.epochs):

            # Shuffle ids, U and H
            np.random.shuffle(ids)
            for j in range(self.S):
                # Update H for each source
                self.H_r[self.class_ids[j],:] *= H_update(V,self.W,self.H_r, source = j)

            # Update W for each batch
            for b in range(N_batches):

                batch_ids = ids[np.arange(b*self.batch_size,(b+1)*self.batch_size)%self.N_r]

                for j in range(self.S):
                
                    # Makes copies of the shuffled data
                    V_batch = V[:,batch_ids]
                    H_batch = self.H_r[:,batch_ids]

                    # Update W for each batch
                    self.W[:,self.class_ids[-1]] *= W_update(V_batch, self.W, H_batch)

            if self.normalize:
                norms = np.linalg.norm(self.W, axis = 0)
                self.W = self.W/(norms + 1e-10)
                self.H_r = norms[:,np.newaxis] * self.H_r

    # Semi-supervised and adversarial fitting, never used.
    #def fit_adv_semi(self, V, U_z):
    #    """
    #    Semi supervised fitting. Assumes self.W is initialized already
    #    """

    #    self.M = V.shape[0]
    #    self.N_r = V.shape[1]
    #    self.N_z = U_z.shape[1]
    #    self.S = len(self.ds)

    #    if self.batch_size == None:
    #        self.batch_size = self.N_r
    #    if self.batch_size_z == None:
    #        self.batch_size_z = self.N_z

    #    N_batches_r = self.N_r//self.batch_size
    #    N_batches_z = self.N_z//self.batch_size_z  
    #    
    #    # Used for inner epoch loop
    #    if self.true_sample == "std":
    #        N_batches = N_batches_r
    #    elif self.true_sample == "adv":
    #        N_batches = N_batches_z
    #    else:
    #        N_batches = np.minimum(N_batches_r, N_batches_z)

    #    self.initH(V = V, prob = "semi_adv")

    #    # List of ids that will be shuffled
    #    ids_r = np.arange(0,self.N_r)
    #    ids_z = np.arange(0,self.N_z)

    #    # Define updates and loss func, leftover from old code
    #    W_update = self.adv_W_semi_update

    #    for i in range(self.epochs):

    #        # Shuffle ids, U and H
    #        np.random.shuffle(ids_r)
    #        np.random.shuffle(ids_z)

    #        for b in range(N_batches):

    #            batch_ids_r = ids_r[np.arange(b*self.batch_size,(b+1)*self.batch_size)%self.N_r]
    #            batch_ids_z = ids_z[np.arange(b*self.batch_size_z,(b+1)*self.batch_size_z)%self.N_z]

    #            for j in range(self.S):

    #                # Update H for each source
    #                self.H_r[self.class_ids[j],:] *= self.H_semi_update(V,self.W,self.H_r, source = j)
    #            
    #            self.H_z *= self.H_update(U_z,self.W[:,self.class_ids[-1]],self.H_z)
    #            
    #            # Makes copies of the shuffled data
    #            V_batch = V[:,batch_ids_r]
    #            U_z_batch = U_z[:,batch_ids_z]
    #            H_r_batch = self.H_r[:,batch_ids_r]
    #            H_z_batch = self.H_z[:,batch_ids_z]

    #            # Update W for each batch
    #            self.W[:,self.class_ids[-1]] *= W_update(V_batch, U_z_batch, self.W, H_r_batch, H_z_batch)

    #        if self.normalize:
    #            norms = np.linalg.norm(self.W, axis = 0)
    #            self.W = self.W/(norms + 1e-10)
    #            self.H_r = norms[:,np.newaxis] * self.H_r
    #            self.H_z = norms[np.sum(self.ds[:-1]),np.newaxis] * self.H_z
            
            
                
    def transform(self, U, current = False, WtW = None, WtU = None):
        """
        Solves
        H(U,W) = arg min_{H \ge 0} \|U - WH\|_F^2

        Current uses H_r that was used in fitting.
        """ 
        if current:
            return self.H_r
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

            H = H * WtU/(np.dot(WtW, H) + self.mu_H)

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
        tau_A = 0.1, tau_S = 0.5, warm_start_epochs = 0, **NMF_args):
  
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
        self.tau_S = tau_S
        
        if NMFs is None:
            self.NMFs = []
            for i,d in enumerate(ds):
                self.NMFs.append(NMF(d, ds = ds, tau_A = self.tau_A[i], tau_S = tau_S, prob = prob,
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
            H_concat = self.NMF_concat.H_r
        else:  
            H_concat = self.NMF_concat.transform(V)

        # Pixel, source, data
        self.Us = np.zeros((V.shape[0], self.S, V.shape[1]))

        for i in range(self.S):
            H = H_concat[sum(self.ds[:i]):sum(self.ds[:i+1])]
            self.Us[:,i,:] = np.dot(self.NMFs[i].W, H)
        
        if self.wiener == True:
            U_sum = np.sum(self.Us, axis = 1) + self.eps
            for i in range(self.S):
                self.Us[:,i,:] = V * self.Us[:,i,:]/U_sum
        return self.Us
    
    def fit(self, U_r = None, V = None, U_sup = None, V_sup = None):
        """
        Fits NMF bases for all sources for all different problem settings

        Input:
            U_r: list of weak supervision datasets for each source
            V: Weak supervision mixed data
            U_sup: list of strong supervision datasets for each source
            V: Strong supervision mixed data so that the i-th data corresponds
                to the i-th data in each dataset in U_sup.
        """

        if self.prob == "std":

            for i,nmf in enumerate(self.NMFs):
                if U_sup is not None and U_r is not None:
                    nmf.fit_std(np.concatenate((U_r[i], U_sup[i]), axis = -1))
                else:
                    nmf.fit_std(U_r[i])
            self.to_concat()
        
        elif self.prob == "adv":

            if U_sup is not None and V_sup is not None and U_r is not None:
                V_ = np.concatenate((V, V_sup), axis = -1)
                U_ = []
                for i in range(self.S):
                    U_.append(np.concatenate((U_r[i],U_sup[i]), axis = -1))
            else:
                U_ = U_r
                V_ = V

            U_z = self.create_adversarial(U_, V = V_)
            for i,nmf in enumerate(self.NMFs):
                if self.warm_start_epochs > 0:
                    nmf.prob = "std"
                    nmf.epochs = self.warm_start_epochs
                    nmf.fit_std(U_[i])
                    nmf.epochs = self.epochs
                    nmf.prob = "adv"
                nmf.fit_adv(U_[i], U_z[i])
            self.to_concat()
        
        elif self.prob == "exem":

            for i,nmf in enumerate(self.NMFs): 
                if U_sup is not None:
                    nmf.fit_exem(np.concatenate((U_r[i], U_sup[i]), axis = -1))
                else:
                    nmf.fit_exem(U_r[i])
            self.to_concat()

        elif self.prob == "sup":

            assert U_sup is not None and V_sup is not None, "Discriminative fitting requires U_sup and V_sup, but at least one is None"

            if self.warm_start_epochs > 0:
                for i,nmf in enumerate(self.NMFs):
                    nmf.prob = "std"
                    nmf.epochs = self.warm_start_epochs
                    nmf.fit_std(U_r[i] if U_r is not None else U_sup[i])
                    nmf.epochs = self.epochs
                    nmf.prob = "sup"
                self.to_concat()
            self.NMF_concat.fit_sup(U_sup, V_sup)
            self.from_concat()

        elif self.prob == "full":

            assert U_r is not None and U_sup is not None and V_sup is not None, "Full fitting requires U_r, U_sup and V_sup"

            U_z = self.create_adversarial(U_r, V = V)

            if self.warm_start_epochs > 0:
                for i,nmf in enumerate(self.NMFs):
                    nmf.prob = "std"
                    nmf.epochs = self.warm_start_epochs
                    nmf.fit_std(U_r[i])
                    nmf.epochs = self.epochs
                    nmf.prob = "full"
                self.to_concat()
            self.NMF_concat.fit_full(U_r, U_z, U_sup, V_sup)
            self.from_concat()
        
        elif self.prob == "semi":
            for i,nmf in enumerate(self.NMFs):
                if i < len(self.ds) - 1:
                    nmf.prob = "std"
                    nmf.fit_std(U_r[i])
                else:
                    nmf.M = V.shape[0]
                    nmf.N_r = V.shape[1]
                    if nmf.W is None:
                        nmf.init = "rand"
                        nmf.initW()
            self.to_concat() 
            self.NMF_concat.fit_std_semi(V = V)
            self.from_concat()
        
        elif self.prob == "semi_adv":
            U_z = self.create_adversarial(U_r, V = V)
            for i,nmf in enumerate(self.NMFs):
                if i < len(self.ds) - 1:
                    nmf.prob = "adv"
                    nmf.fit_adv(U_r[i], U_z[i])
                else:
                    nmf.M = V.shape[0]
                    nmf.N_r = V.shape[1]
                    if nmf.W is None:
                        nmf.init = "rand"
                        nmf.initW()
            self.to_concat() 
            #self.NMF_concat.fit_adv_semi(V = V, U_z = U_z[-1])
            self.NMF_concat.fit_std_semi(V = V)
            self.from_concat()

        elif self.prob == "semi_exem":
            for i,nmf in enumerate(self.NMFs):
                if i < len(self.ds) - 1:
                    nmf.prob = "exem"
                    nmf.fit_exem(U_r[i])
                else:
                    nmf.init = "rand"
                    nmf.M = V.shape[0]
                    nmf.N_r = V.shape[1]
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

    def create_adversarial(self, U_r, V = None):
        """
        Generates adversarial dataset that potentially contains both mixed data and data from other sources
        U_{Z_i} = tau_i * [sqrt(omega_i1 N_z/N_1)U_1 ... sqrt((1- sum_{j \neq i} omega_{ij}) N_z/N_V ) beta_i V]

        Input:
            U_r: List of S (M,N) numpy arrays containing true data of each source
            V: (M,N) numpy array containing mixed data
            taus: List of S floats. Controls the overall weight of the adversarial term. Defaults to $1$
            betas: List of S floats. Controlls the weight of the mixed data. Defaults to 1.
            omegas: 2-D list of (S-1)**2 floats. Controls the weights of the individual sources. Defaults to N_i/\hat{N}_i
        Output:
            U_Z: List of S arrays each of size (M,N_{Z_i})
        """
        U_Z = []
        Ns = []
        useMixed = (V is not None)
        S = len(U_r)
        for i in range(S):
            Ns.append(U_r[i].shape[1])

        N_Zs = []
        # All my homies hate list comprehension
        for i in range(S):
            N_Zs.append(np.sum(Ns[:i] + Ns[i+1:]))
            if useMixed:
                N_Zs[i] += V.shape[1]

        if self.betas is None:
            self.betas = [1.0] * (S + int(useMixed))

        if self.omegas is None:
            self.omegas = np.zeros((S, S-1))

            # For each source we need to create a dataset 
            for i in range(S):

                # We need to account for the weight of the reminaing S-1 sources after taking one source out
                for j in range(S - 1):
                    self.omegas[i,j] = Ns[j + (j >= i)]/N_Zs[i]

        for i in range(len(U_r)):
            U_r_ = U_r[:i] + U_r[i+1:]
            for j in range(len(U_r_)):
                U_r_[j] = np.sqrt(self.omegas[i,j] * N_Zs[i]/Ns[j + (j >= i)]) * U_r_[j]
            if useMixed:
                U_r_.append(np.sqrt((1.0 - np.sum(self.omegas[i])) * N_Zs[i]/V.shape[1]) * self.betas[i] * V)

            if self.prob == "adv" or self.prob == "semi_adv":
                U_Z.append(np.copy(np.sqrt(self.tau_A[i]) * np.concatenate(U_r_, axis = 1)))

            elif self.prob == "full":
                U_Z.append(np.copy(np.sqrt((1.0 - self.tau_S) * self.tau_A[i]) * np.concatenate(U_r_, axis = 1)))

        return U_Z

class audio_separation:
    """
    Wrapper class for NMF_separation for audio applciations.
    Main part is applying STFT and ISTFT and metrics for SNR and SDR

    Parameters
    --------
        prob: str, problem formulation. Currently only supports "semi", for standard semi-supervised fitting
            and "semi-adv" for adversarial semi-supervised fitting.
        project: bool, wether or not to use a a basis for the unobserved data during testing
        n_fft: int, size of fft used for STFT. Should be a power of 2.
        hop_length: int, hop length passed to STFT
        win_length: int, window length passed to STFT
        sep_args: arguments passed to NMF_separation
    """

    def __init__(self,prob = "semi", project = False, n_fft = 512, hop_length = None, win_length = None, **sep_args):
        self.prob = prob
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.project = project

        self.sep = NMF_separation(prob = self.prob, **sep_args)

    def fit(self, u, v = None):
        """
        Fits bases given list of audio signals u and list of audio signals v.
        Currently only really works for semi and semi_adv
        """

        U_speech = []
        Z_speech = []
        for i in range(len(u)):
            Z_speech.append(stft(u[i],n_fft=self.n_fft, hop_length = self.hop_length, win_length = self.win_length))
            U_speech.append(np.abs(Z_speech[i]))
        U_speech_concat = np.concatenate(U_speech, axis = -1)

        V = []
        Z_mix =  []
        for i in range(len(v)):
            Z_mix.append(stft(v[i],n_fft=self.n_fft, hop_length = self.hop_length, win_length = self.win_length))
            V.append(np.abs(Z_mix[i]))
        V_concat = np.concatenate(V, axis = -1)

        self.sep.fit(U_r = [U_speech_concat], V = V_concat)


    def separate(self, v = None):
        """
        Separates list of audio signals v. 
        """ 
        out = []
        for i in range(len(v)):
            Z = stft(v[i],n_fft=self.n_fft, hop_length = self.hop_length, win_length = self.win_length)
            if self.project:
                U_reconst = np.dot(self.sep.NMFs[0].W, self.sep.NMFs[0].transform(np.abs(Z)))     
            else:
                U_reconst = self.sep.separate(np.abs(Z))[:,0,:]
            out.append(istft(U_reconst * np.exp(1j * np.angle(Z)), n_fft = self.n_fft, length = len(v[i])))

        return out

    def eval(self, u, v, metric = 'SNR', out = None, aggregate = None):
        """
        Evaluates difference between list of audio signals u and list of audio signals v. 
        """
        
        assert len(u) == len(v)
        
        if out is None:
            out = self.separate(v)

        if metric == 'SNR':
            return np.mean([calculate_snr(u[i], out[i]) for i in range(len(v))])
        elif metric == 'SDR':
            #return np.nanmean(np.concatenate([bss_eval(u[i], out[i], window = self.sdr_window, hop = self.sdr_window)[0] for i in range(len(u))], axis = -1))
            return np.mean([calculate_sdr(u[i], out[i]) for i in range(len(u))])


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

    def fit(self, U_r = None, V = None, U_sup = None, V_sup = None, refit = False):
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
                    train_V = V_sup[:, train_idx]
                    test_V = V_sup[:, start:end]

                    sep = self.method(**params)

                    sep.fit(U_r = U_r, V = V, U_sup = train_U, V_sup = train_V)
                    dists = sep.eval(test_U, test_V, metric = self.metric, aggregate = self.source_aggregate)
                    if self.data_aggregate == "mean":
                        ex_val += np.mean(dists) / self.cv
                    elif self.data_aggregate == "median":
                        ex_val += np.median(dists) / self.cv
            else:
                sep = self.method(**params)
                sep.fit(U_r = U_r, V = V, U_sup = U_sup, V_sup = V_sup)

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
            self.best_model.fit(U_r = U_r, V = V, U_sup = U_sup, V_sup = V_sup)

        if self.verbose:
            print("Best param", self.best_param, self.best_val)
        return results
