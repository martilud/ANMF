import numpy as np
from utils import *
from copy import deepcopy

class NMF:

    def __init__(self, d = None, ds = None, tau = None,
        W = None, loss = "square", prob = "std", init = "exem",
        epochs = 50, test_epochs = 50, warm_start_epochs = 50,
        batch_size = 250, batch_size_z = 250, batch_size_sup = 250,   
        mu_W = 1e-6, mu_H = 1e-6, verbose = False):
        """
        Class for Non-Negative Matrix Factorization for source separation. 
        Decomposes Non-negative data stored columnwise in and m times n array U as U approx WH,
        where W is m times d and H is d times n.

        Class has several fit functions for different variants of NMF.

        TODO:
        - Write code for full NMF
        - Implement argument for which dataset to oversample/undersample before full NMF
            -> Do this with one argument "true_sample" which decides which dataset we will sample fully, that is, how many batches we go through
            -> For all other datasets, evaluate the i%N_batches(for that class) batch, so that we undersample and oversample automatically
        - Remove everything that is not squared 
        - Write proper docstrings
        - Write proper convergence
        
        input:
            loss: str, '2' for 2-norm, 'fro' for frobenius norm and 'square' for squared 2-norm/frobenius norm
            d: int, amount of basis vectors in W
        """
        self.loss = loss
        self.mu_W = mu_W
        self.mu_H = mu_H
        self.prob = prob
        self.init = init
        self.epochs = epochs
        self.test_epochs = test_epochs
        self.warm_start_epochs = warm_start_epochs
        self.d = d
        self.ds = ds
        self.verbose = verbose
        self.batch_size = batch_size
        self.batch_size_z = batch_size_z
        self.batch_size_sup = batch_size_sup
        self.tau = tau

        # Copy W if it is given
        self.W = W

        # Initialize H_r and H_z
        self.H_r = None
        self.H_z = None

        # Lengths
        self.M = None
        self.N_r = None
        self.N_v = None

        # Number of sources
        self.S = None

    def std_W_update_square(self,U_r,W,H_r):
        """
        Update rule for W
        """
        # Is W(HH^T) always faster than (WH)H^T? Depends on batch size and d. 
        H_rH_rT = np.dot(H_r, H_r.T)
        #WH_r = np.dot(W,H_r)
        invN_r = U_r.shape[1]
        W_update = np.dot(U_r, H_r.T) * invN_r / (np.dot(W, H_rH_rT) * invN_r + self.mu_W)
        #W_update = np.dot(U_r, H_r.T) / (np.dot(W, H_rH_rT) + self.mu_W)
        #W_update = np.dot(U_r, H_r.T) / (np.dot(WH_r, H_r.T) + self.mu_W)
        return W_update
    
    def std_W_update_2(self,U_r,W,H_r):
        WH_r = np.dot(W,H_r)
        norms = np.linalg.norm(U_r - WH_r, axis = 0)
        H_bar_T = np.divide(H_r, norms).T
        W_update = np.dot(U_r, H_bar_T) / (np.dot(WH_r, H_bar_T) + self.mu_W)
        return W_update   

    def adv_W_update_square(self, U_r, U_z, W, H_r, H_z):

        WH_r = np.matmul(W,H_r)
        WH_z = np.matmul(W,H_z)
        invN_r = 1.0/U_r.shape[1]
        invN_v = 1.0/U_z.shape[1]
        W_update = (np.dot(WH_z, H_z.T) * invN_v + np.dot(U_r,H_r.T) * invN_r) \
        /(np.dot(WH_r, H_r.T)*invN_r + np.dot(U_z,H_z.T)*invN_v + self.mu_W)
        return W_update

    def adv_W_update_fro(self, U_r, U_z, W, H_r, H_v):
        WH_r = np.matmul(W,H_r)
        WH_v = np.matmul(W,H_v)
        inv_norm_r = 1.0/(np.linalg.norm(WH_r - U_r, 'fro') * U_r.shape[1])
        inv_norm_v = 1.0/(np.linalg.norm(WH_v - U_z, 'fro') * U_z.shape[1])
        W_update = (inv_norm_v * np.dot(WH_v, H_v.T) + inv_norm_r * np.dot(U_r,H_r.T)) \
        /(inv_norm_r * np.dot(WH_r, H_r.T) + inv_norm_v * np.dot(U_z,H_v.T) + self.mu_W)
        return W_update
    
    def adv_W_update_2(self,U_r,U_z,W,H_r,H_v):
        WH_r = np.dot(W,H_r)
        WH_v = np.dot(W,H_v)
        norms_r = np.linalg.norm(U_r - WH_r, axis = 0) * U_r.shape[1] 
        norms_v = np.linalg.norm(U_z - WH_v, axis = 0) * U_z.shape[1]
        H_bar_rT = np.divide(H_r, norms_r).T
        H_bar_vT = np.divide(H_v, norms_v).T
        W_update = (np.dot(WH_v, H_bar_vT) + np.dot(U_r,H_bar_rT)) \
        /(np.dot(WH_r, H_bar_rT) + np.dot(U_z,H_bar_vT) + self.mu_W)
        return W_update
    
    def H_update(self,U,W,H, WtU = None, WtW = None):
        """
        Update rule for H
        """
        if WtU is None:
            WtU = np.dot(W.T, U)
        if WtW is None:
            WtW = np.dot(W.T, W)
        H_update = WtU/(np.dot(WtW, H) + self.mu_H)
        return H_update

    def H_update_2(self,U,W,H, WtU = None, WtW = None):
        if WtU is None:
            WtU = np.dot(W.T,U)
        if WtW is None:
            WtW = np.dot(W.T,W)
        WH = np.dot(W,H)
        norms = np.linalg.norm(U - WH, axis = 0)
        WtU_bar = WtU/norms
        H_bar = H/norms
        H_update = WtU_bar/(np.dot(WtW, H_bar) + self.mu_H)
        return H_update

    def initializeWH(self, U_r = None, U_z = None, U_sup = None, V_sup = None, prob = "std",
        M = None, N_r = None, N_v = None , d = None, ds = None):
        """
        Initialize W and H for fitting
        Input:

            prob: string. Problem type. 'std' for standard NMF, 'adv' for adversarial, 'sup' for supervised
            type: string. Type of initialization. 'rand' for random, 'exem' for exemplar based, which means we also have to pass U_r
        """

        if M is None:
            M = self.M
        
        if N_r is None:
            N_r = self.N_r
        
        if N_v is None:
            N_v = self.N_v
        
        if d is None:
            d = self.d
            if prob == "sup":
                d = np.sum(ds)

        if self.W is None:
            if self.init == "rand":
                self.W = np.random.uniform(0,1,(M,d))
            elif self.init == "exem":
                if prob != "sup":
                    self.W = U_r[:,np.random.choice(U_r.shape[1], size = d, replace = False)]
                else:
                    self.W = np.zeros((M,np.sum(ds)))
                    for j in range(len(ds)):
                        self.W[:,sum(ds[:j]):sum(ds[:j+1])] = U_sup[j][:,np.random.choice(U_sup[j].shape[1], size = ds[j], replace = False)]

        if self.init == "rand":
            # Redundantly check if H_r exists
            self.H_r = np.random.uniform(0,1,(d,N_r))
            if prob == "adv":
                self.H_z = np.random.uniform(0,1,(d,N_v))
        elif self.init == "exem":
            if prob != "sup":
                _, self.H_r = self.transform(U_r)
                if prob == "adv":
                    _, self.H_z = self.transform(U_z) 
            else:
                _, self.H_z = self.transform(V_sup)

    def std_loss_2(self,U_r,WH_r):
        return np.mean(np.linalg.norm(U_r - WH_r,2, axis = 0))
    
    def std_loss_fro(self,U_r,WH_r):
        return 1/U_r.shape[1] * np.linalg.norm(U_r - WH_r, 'fro')

    def std_loss_square(self,U_r, WH_r):
        return 1/U_r.shape[1] * np.linalg.norm(U_r - WH_r, 'fro')**2
    
    def adv_loss_2(self,U_r,U_z,WH_r,WH_v):
        return np.mean(np.linalg.norm(U_r - WH_r,2,axis = 0)) - np.mean(np.linalg.norm(U_z - WH_v,2,axis = 0))

    def adv_loss_fro(self,U_r,U_z,WH_r,WH_v):
        return 1/U_r.shape[1] * np.linalg.norm(U_r - WH_r, 'fro') - 1/U_z.shape[1] * np.linalg.norm(U_z - WH_v, 'fro')
    
    def adv_loss_square(self,U_r,U_z,WH_r,WH_v):
        return 1/U_r.shape[1] * np.linalg.norm(U_r - WH_r, 'fro')**2 - self.tau/U_z.shape[1] * np.linalg.norm(U_z - WH_v, 'fro')**2

    def fit_std(self, U_r,update_H = False, conv = False):
        """
        Fits standard NMF by solving

        min_{W \ge 0} 1/N \|U - WH(U)\|_F^2
        where H(U) = arg min_{H \ge 0} \|U - WH\|_F^2

        Fitting is done using a mini-batch multiplicative algorithm that is initialization sensitive
        """
        U_r_ = np.copy(U_r)
        self.M = U_r_.shape[0]
        self.N_r = U_r_.shape[1]
        
        # Calculate Number of Batches
        N_batches = self.N_r//self.batch_size 

        # Initialize if nothing exists
        if self.W is None or self.H_r is None:
            self.initializeWH(U_r = U_r_, prob = "std")

        # List of ids that will be shuffled
        ids = np.arange(0,self.N_r)

        if self.loss == '2' or self.loss == 2:
            W_update = self.std_W_update_2
            H_update = self.H_update_2
        elif self.loss == 'fro':
            W_update = self.std_W_update_square
            H_update = self.H_update
        elif self.loss == 'square':
            W_update = self.std_W_update_square
            H_update = self.H_update

        if conv:
            if self.loss == '2' or self.loss == 2:
                loss_func = self.std_loss_2
            elif self.loss == 'fro':
                loss_func = self.std_loss_fro
            elif self.loss == 'square':
                loss_func = self.std_loss_square
        
        if conv or self.verbose:
            loss_std = np.zeros((self.epochs + 1))
            WH_r = np.dot(self.W,self.H_r)
            loss_std[0] = loss_func(U_r_,WH_r)

        for i in range(self.epochs):
            # Shuffle ids, U and H
            np.random.shuffle(ids)
            U_r_ = U_r_[:,ids]
            self.H_r = self.H_r[:,ids] 

            # Split into batches
            U_r_b = np.split(U_r_, N_batches, axis = 1)
            H_r_b = np.split(self.H_r, N_batches, axis = 1) 

            for b in range(N_batches):

                # Update W for each batch
                self.W = self.W * W_update(U_r_b[b], self.W, H_r_b[b])

                # Calculate H update and update H
                if update_H:
                    H_r_b[b] = H_r_b[b] * H_update(U_r_b[b], self.W, H_r_b[b])
                
            if update_H:
                self.H_r = np.concatenate(H_r_b, axis = 1)
            else:
                self.H_r = self.H_r * H_update(U_r_,self.W,self.H_r)

            if self.verbose or conv:
                WH_r = np.dot(self.W,self.H_r)
                loss_std[i+1] = loss_func(U_r_,WH_r)
                if self.verbose:
                    print(f"Epoch: {i+1}, Loss: {loss_std[i+1]}")
        if conv:
            return loss_std

    def fit_adv(self, U_r, U_z, update_H = False, conv = False):
        """
        Fits adversarial NMF by solving

        min_{W \ge 0} 1/N \|U - WH(U)\|_F^2 - \tau/\hat{N} \|U - WH(\hat{U})\|_F^2 
        where H(U) = arg min_{H \ge 0} \|U - WH\|_F^2

        Here U is true data and \hat{U} is adversarial data.

        Fitting is done using a mini-batch multiplicative algorithm that is initialization sensitive
        """
        U_r_ = np.copy(U_r)
        self.M = U_r_.shape[0]
        self.N_r = U_r_.shape[1]
        U_z_ = np.copy(U_z)
        assert self.M == U_z_.shape[0], f"U_r has first axis {self.M} and U_z has first axis {U_z.shape[0]} which does not match."
        self.N_v = U_z_.shape[1]

        # Calculate Number of Batches
        N_batches_r = self.N_r//self.batch_size
        N_batches_v = self.N_v//self.batch_size_z  
        
        # Used for inner epoch loop
        N_batches = np.minimum(N_batches_r, N_batches_v)

        if self.W is None or self.H_r is None or self.H_v is None:
            self.initializeWH(U_r = U_r, U_z = U_z, prob = "adv")

        # List of ids that will be shuffled
        ids_r = np.arange(0,self.N_r)
        ids_v = np.arange(0,self.N_v)

        # Set update
        if self.loss == "square":
            W_update = self.adv_W_update_square
            H_update = self.H_update
        elif self.loss == "fro":
            W_update = self.adv_W_update_fro
            H_update = self.H_update
        elif self.loss == '2' or self.loss == 2:
            W_update = self.adv_W_update_2
            H_update = self.H_update_2

        if conv == True:
            if self.loss == '2' or self.loss == 2:
                loss_func_std = self.std_loss_2
                loss_func_adv = self.adv_loss_2
            elif self.loss == 'fro':
                loss_func_std = self.std_loss_fro
                loss_func_adv = self.adv_loss_fro
            elif self.loss == 'square':
                loss_func_std = self.std_loss_square
                loss_func_adv = self.adv_loss_square
        
        if conv or self.verbose:
            loss_std = np.zeros((self.epochs + 1))
            loss_adv = np.zeros((self.epochs + 1))
            WH_r = np.dot(self.W,self.H_r)
            WH_z = np.dot(self.W,self.H_z)
            loss_std[0] = loss_func_std(U_r_,WH_r)
            loss_adv[0] = loss_func_adv(U_r_,U_z_, WH_r, WH_z)
            

        for i in range(self.epochs):
            # Shuffle ids, U and H
            np.random.shuffle(ids_r)
            np.random.shuffle(ids_v)
            U_r_ = U_r_[:,ids_r]
            self.H_r = self.H_r[:,ids_r]
            U_z_ = U_z_[:,ids_v]
            self.H_z = self.H_z[:,ids_v]  

            # Split into batches
            U_r_b = np.split(U_r_, N_batches_r, axis = 1)
            H_r_b = np.split(self.H_r, N_batches_r, axis = 1)
            U_z_b = np.split(U_z_, N_batches_v, axis = 1)
            H_z_b = np.split(self.H_z, N_batches_v, axis = 1) 

            for b in range(N_batches):

                # Update W for each batch
                self.W = self.W * W_update(U_r_b[b], U_z_b[b], self.W, H_r_b[b], H_z_b[b])

                if update_H:
                    WtW = np.dot(self.W.T, self.W)
                    H_r_b[b] = H_r_b[b] * H_update(U_r_b[b],self.W,H_r_b[b], WtW = WtW)
                    H_z_b[b] = H_z_b[b] * H_update(U_z_b[b],self.W,H_z_b[b], WtW = WtW)

            if update_H:
                self.H_r = np.concatenate(H_r_b, axis = 1)
                self.H_z = np.concatenate(H_z_b, axis = 1)
            else:
                WtW = np.dot(self.W.T, self.W)
                self.H_r = self.H_r * H_update(U_r_,self.W, self.H_r, WtW = WtW) 
                self.H_z = self.H_z * H_update(U_z_,self.W, self.H_z, WtW = WtW)

            if self.verbose or conv:
                WH_r = np.dot(self.W,self.H_r)
                WH_z = np.dot(self.W,self.H_z)
                loss_std[i+1] = loss_func_std(U_r_,WH_r)
                loss_adv[i+1] = loss_func_adv(U_r_,U_z_, WH_r, WH_z)
                if self.verbose:
                    print(f"Epoch: {i+1}, Loss: {loss_adv[i+1]}")
        if conv:
            return loss_std, loss_adv

    def fit_sup(self,U_r,V,conv = False):
        """
        Solves:

        \min_{W \ge 0} \sum_i^{S} \|U_i - W_i H_i\|_F^2
        s.t H = \argmin_{H} \|V - WH\|_F^2,
        where W = [W_1,...W_S] and H = [H_1,... W_S] are concatenated versions of W and H

        Right now function takes in U_r as a list of (M,N) arrays, it should handle a (M,S,N) array too. 
        input:
           U_r: (M,S,N) array
        """
        assert len(U_r) == len(self.ds), "U_r and self.ds shapes do not match"
        #self.Ms = np.zeros(len(U_r), dtype = int)

        U_r_ = []
        for i in range(len(U_r)):
            U_r_.append(np.copy(U_r[i])) 

        self.N_r = V.shape[1] 
        self.M = V.shape[0]
        V_ = np.copy(V)


        self.S = len(U_r)
        
        # Calculate Number of Batches
        self.N_batches = self.N_r//self.batch_size_sup 

        if self.loss == '2' or self.loss == 2:
            update = self.std_W_update_2
        elif self.loss == 'fro':
            update = self.std_W_update_square
        elif self.loss == 'square':
            update = self.std_W_update_square
            
        if self.W is None or self.H_r is None:
            self.initializeWH(U_sup = U_r_, V_sup = V_, ds = self.ds, prob = "sup")

        N_ids = np.arange(0,self.N_r)

        if conv:
            loss_func = self.std_loss_square
            loss_sup = np.zeros((self.S, self.epochs + 1))
            for j in range(self.S):
                WH_r = np.dot(self.W[:,sum(self.ds[:j]):sum(self.ds[:j+1])], self.H_r[sum(self.ds[:j]):sum(self.ds[:j+1]),:])
                loss_sup[j,0] = loss_func(U_r_[j], WH_r)

        for i in range(self.epochs):

            # Shuffle data
            np.random.shuffle(N_ids)
            V_ = V_[:,N_ids]
            self.H_r = self.H_r[:,N_ids]

            # Iterate over each source
            for j in range(self.S):

                U_r_[j] = U_r_[j][:,N_ids]

                # Split into batches
                U_r_b = np.split(U_r_[j], self.N_batches, axis = 1)
                H_r_b = np.split(self.H_r[sum(self.ds[:j]):sum(self.ds[:j+1]),:], self.N_batches, axis = 1) 

                for b in range(self.N_batches):

                    # For each batch, calculate update
                    W_update = update(U_r_b[b], self.W[:,sum(self.ds[:j]):sum(self.ds[:j+1])], H_r_b[b])

                    # Update W
                    self.W[:,sum(self.ds[:j]):sum(self.ds[:j+1])] = self.W[:,sum(self.ds[:j]):sum(self.ds[:j+1])] * W_update

            self.H_r = self.H_r * self.H_update(V_,self.W,self.H_r)
            
            if conv:
                for j in range(self.S):
                    WH_r = np.dot(self.W[:,sum(self.ds[:j]):sum(self.ds[:j+1])], self.H_r[sum(self.ds[:j]):sum(self.ds[:j+1]),:])
                    loss_sup[j,i+1] = loss_func(U_r_[j], WH_r)
        if conv:
            return loss_sup

    def fit_exem(self, U_r = None):
        self.initializeWH(U_r = U_r)

    
    def fit_full(self, U_r, U_z, U_sup, V_sup):
        """
        Fit function that can handle fitting for fitting FNMF which includes weak supervision data, adversarial data
        and strong supervision data. Thus it can be used both to fit a single (A)NMF as well as S FNMFs together

        """
        U_r_ = np.copy(U_r)
        U_z_ = np.copy(U_z)
        U_sup = np.copy(U_sup)
        V_sup = np.copy(V_sup)


        # Initialize W and the different latent variables
        self.initializeWH(U_r = U_r, U_z = U_z, U_sup = U_sup, V_sup = V_sup)

        Nrs = []
        Nzs = []
        for i in range(len(U_r_)):
            Nrs.append(U_r_[i].shape[1])
            Nzs.append(U_z_[i].shape[1])

        # Calculate number of batches
        N_batches_r = np.min(Nrs)//self.batch_size
        N_batches_v = np.min(Nzs)//self.batch_size_z 
        N_batches_sup = U_sup[0]//self.batch_size_sup
        
        # Used for inner epoch loop
        N_batches = np.minimum(N_batches_r, N_batches_v, N_batches_sup)
        
        # Start main loop

        ids_r = np.arange(0,self.N_r)
        ids_v = np.arange(0,self.N_v)
        for i in range(self.epochs):
            return 0
            # Shuffle data

 
            # Update W for each source
        #    for j in range(self.S):

                # Split data into batches

                # For each batch, calculate updates and update
            #        for b in range(self.N_batches):
            #        i
            
            # Update different Hs
                
    def transform(self, U, WtW = None, WtU = None):
        """
        
        """ 
        if WtW is None:
            WtW = np.dot(self.W.T, self.W)
        if WtU is None:
            WtU = np.dot(self.W.T,U)
        N = U.shape[1]
        H = np.random.uniform(0,1,(self.d,N))

        if self.loss == '2' or self.loss == 2:
            H_update = self.H_update_2
        elif self.loss == 'fro':
            H_update = self.H_update
        elif self.loss == 'square':
            H_update = self.H_update

        for i in range(self.test_epochs):

            H = H*H_update(U, self.W, H, WtW = WtW, WtU = WtU)

            #if self.verbose:
                #print(f"Iter: {i}, Loss: {1/N * np.linalg.norm(U - np.dot(self.W, H),'fro')**2}")
        return np.dot(self.W,H), H

class NMF_separation:
    """
    Class for source separation with NMF

    TODO:
        - Make sure that all relevant arguments are passed to NMF
        - Fit for full NMF
        - Deciede how to handle the eval function.
            -> Should it return a metric for each source, or should we do some sort of broadcast?
            -> This function might need to have an axis argument too
    """
    def __init__(self, ds = None, mu_W = 1e-6, mu_H = 1e-6,
    epochs = 25, warm_start_epochs = 25, test_epochs = 25,
    prob = "std", loss = "square", init = "exem",
    batch_size_r = 250, batch_size_z = 250, batch_size_sup = 250,
    wiener = True, eps = 1e-10,
    taus = None, betas = None, omegas = None,
    NMFs = None, verbose = False):
        self.ds = ds
        self.loss = loss
        self.prob = prob
        self.init = init
        self.mu_W = mu_W
        self.mu_H = mu_H
        self.epochs = epochs
        self.test_epochs = test_epochs
        self.N_sources = len(ds)
        self.wiener = wiener
        self.warm_start_epochs = warm_start_epochs
        self.eps = eps
        self.taus = taus
        self.betas = betas
        self.omegas = omegas

        if NMFs is None:
            self.NMFs = []
            for i,d in enumerate(ds):
                self.NMFs.append(NMF(d, ds = ds, tau = taus[i] if taus is not None else None, loss = loss, epochs = epochs,
                    test_epochs = test_epochs, init = init, prob = prob,
                    batch_size_r = batch_size_r, batch_size_z = batch_size_z, batch_size_sup = batch_size_sup,
                    mu_W = mu_W, mu_H = mu_H, verbose = verbose))
        else:
            self.NMFs = NMFs

        self.NMF_concat = NMF(d = sum(ds), ds = ds, loss = loss, epochs = epochs, test_epochs = test_epochs,
            init = init, prob = prob,
            batch_size_r = batch_size_r, batch_size_z = batch_size_z, batch_size_sup = batch_size_sup,
            mu_W = mu_W, mu_H = mu_H, verbose = verbose)

    def to_concat(self):

        Ws = []

        for i in range(self.N_sources):
            Ws.append(self.NMFs[i].W)
        W_concat = np.concatenate(Ws, axis = 1)
        self.NMF_concat.W = W_concat

    def from_concat(self):
        for i in range(self.N_sources):
            self.NMFs[i].W = self.NMF_concat.W[:,sum(self.ds[:i]):sum(self.ds[:i+1])]

    def separate(self, V):
        
        _, H_concat = self.NMF_concat.transform(V)

        self.Us = np.zeros((V.shape[0], self.N_sources, V.shape[1]))

        for i in range(self.N_sources):
            H = H_concat[sum(self.ds[:i]):sum(self.ds[:i+1])]
            self.Us[:,i,:] = np.dot(self.NMFs[i].W, H)
        
        if self.wiener == True:
            U_sum = np.sum(self.Us, axis = 1) + self.eps
            for i in range(self.N_sources):
                self.Us[:,i,:] = V * self.Us[:,i,:]/U_sum
        return self.Us
    
    def fit(self, U_r = None, V = None, U_sup = None, V_sup = None):
        """
        Input:
            U_r: list of true datasets U_r for each source 
            U_z: list of adversarial datasets U_r for each source 
            V: supervised data so that the i-th data in V is a mix of the i-th data in U_r.
        """

        if self.prob == "std":
            assert U_r is not None, "Standard fitting requires U_r, but U_r is None"

            for i,nmf in enumerate(self.NMFs):
                # print(f"Fitting NMF number {i+1} out of {len(self.NMFs)}.")
                nmf.fit_std(U_r[i])
            self.to_concat()
        
        elif self.prob == "adv":

            assert U_r is not None, "Adverserial fitting requires U_r, but U_r is None"
            
            U_z = self.create_adversarial(U_r, V = V)

            for i,nmf in enumerate(self.NMFs):
                if self.warm_start_epochs > 0:
                    nmf.epochs = self.warm_start_epochs
                    nmf.fit_std(U_r[i])
                    nmf.epochs = self.epochs
                nmf.fit_adv(U_r[i], U_z[i])
            self.to_concat()
        
        elif self.prob == "exem":
            assert U_r is not None or U_sup is not None, "Exemplar-based fitting requires U_r or U_sup, but both are None"

            for i,nmf in enumerate(self.NMFs): 
                nmf.fit_exem(U_r[i] if U_r is not None else U_sup[i])
            self.to_concat()

        elif self.prob == "sup":

            assert U_sup is not None and V_sup is not None, "Discriminative fitting requires U_sup and V_sup, but at least one is None"

            if self.warm_start_epochs > 0:
                for i,nmf in enumerate(self.NMFs):
                    nmf.epochs = self.warm_start_epochs
                    nmf.fit_std(U_sup[i])
                    nmf.epochs = self.epochs
                self.to_concat()
            self.NMF_concat.fit_sup(U_sup, V_sup)
            self.from_concat()
    
    def eval(self, U_test, V_test, metric = 'norm'):
        """
        Tests a fitted method on the test data. Note here that U_test is on a different form compared to 

        To Do: metric should instead pass a suitable function

        input:
            U_test:
            V_test:
            metric: string, 'norm' for squared Frobenius distrance, 'psnr' for PSNR
        """
        out = self.separate(V = V_test)
        if metric == 'norm':
            return np.linalg.norm(out - U_test, axis = 0)**2
        elif metric == 'psnr':
            return PSNR(U_test, out)

    def create_adversarial(self,U_r, V = None):
        """
        Generates adversarial dataset that potentially contains both mixed data and data from other sources
        U_{Z_i} = tau_i * [sqrt(omega_i1 N_z/N_1)U_1 ... sqrt((1- sum_{j \neq i} omega_{ij}) N_z/N_V ) beta_i V]

        Beta should be selected based on the weights of the mix.
        Omegas 
        Input:
            U_r: List of S (M,N_r) numpy arrays containing true data of each source
            U_v: (M,N_V) numpy array containing mixed data
            taus: List of S floats. Controls the overall weight of the adversarial term. Defaults to $1$
            betas: List of S floats. Controlls the weight of the mixed data. Defaults to 1.
            omegas: 2-D list of (S-1)**2 floats. Controls the weights of the individual sources. Defaults to N_i/N_{Z_i}
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

        if self.taus is None:
            self.taus = [1.0] * (S + int(useMixed))

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

            U_Z.append(np.sqrt(self.taus[i]) * np.concatenate(U_r_, axis = 1))
        return U_Z

class random_search:
    """
    Does random search over parameters to try to find optimal solution
    Relevant parameters are (also the initialization is random, so it is technically a parameter):
    d, μH, Test Epochs, μW, Training Epochs, Batch Sizes, Warm Start Epochs, τ, ω 

    Function will evaluate the mean of the metric passed to eval
    """
    def __init__(self,method,param_dicts, N_ex= 50, metric = "psnr", cv = 0):
        """
        Input:
            method, class with fit, separate and eval functions, like NMF_separate
            param_dicts, list of dictionaries. Each dictionary contains:
                'name': string, which is what will be passed to the __init__ function of method.
                'dist': function, returns a candidate we want to search.
                    Distribution should also handle if parameter is discrete or continuous
                    ex1: lambda : np.random.uniform(0.0,1.0)
                    ex2: lambda : np.random.randint(0,10)
                    ex3: lambda : np.random.choice([10,25,50], replace = True)
                    ex4: lambda : [0.5,0.5]
            cv: int, number of cv iterations for supervised data

        """
        self.method = method
        self.param_dicts = param_dicts
        self.N_ex = N_ex
        self.metric = metric
        self.cv = cv

        self.best_model = None
    
    def fit(self, U_r = None, V = None, U_sup = None, V_sup = None):
        """
        Fits
        """

        # Initial best guess
        val = -np.inf

        if self.cv > 1:
            # Get the number of samples along the second axis
            n = U_sup[0].shape[1]
    
            # Calculate the size of each split
            split_size = n // self.cv
        else:
            U_s = np.zeros((U_sup[0].shape[0], len(U_sup), U_sup[0].shape[1]))
            for i in range(len(U_sup)):
                U_s[:,i,:] = U_sup[i]

        # Results should be stored here!
        results = {}

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
                    dists = sep.eval(test_U, test_V, metric = self.metric)
                    ex_val += np.mean(np.mean(dists, axis = -1)) / self.cv
            else:
                sep = self.method(**params)
                sep.fit(U_r = U_r, V = V, U_sup = U_sup, V_sup = V_sup)
                dists = sep.eval(U_s, V_sup, metric = self.metric)

                # Returns array of size S, one metric for each source
                ex_val = np.mean(dists, axis = -1)

                # Takes the mean of that again
                ex_val = np.mean(ex_val)

            if ex_val > val:
                val = ex_val
                best_param = params
                self.best_model = deepcopy(sep)
            print(params, ex_val)
        
        print()
        print("Best param", best_param, val)













            


