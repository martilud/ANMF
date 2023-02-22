import numpy as np
from utils import *
from copy import deepcopy

class NMF:

    def __init__(self, d = None, ds = None, tau_A = 0.1, tau_S = 0.1,
        W = None, loss = "square", prob = "std", init = "exem",
        epochs = 50, test_epochs = 50, warm_start_epochs = 50,
        batch_size = None, batch_size_z = None, batch_size_sup = None,
        true_sample = "std",   
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
        self.true_sample = true_sample
        self.tau_A = tau_A
        self.tau_S = tau_S

        # Copy W if it is given
        self.W = W

        # Initialize H_r and H_z
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

    def std_W_update(self,U_r,W,H_r):
        """
        Update rule for W for standard NMF and discriminative NMF
        """
        # Is W(HH^T) always faster than (WH)H^T? Depends on batch size and d. 
        H_rH_rT = np.dot(H_r, H_r.T)
        #WH_r = np.dot(W,H_r)
        invN_r = U_r.shape[1]
        W_update = np.dot(U_r, H_r.T) * invN_r / (np.dot(W, H_rH_rT) * invN_r + self.mu_W)
        #W_update = np.dot(U_r, H_r.T) / (np.dot(W, H_rH_rT) + self.mu_W)
        #W_update = np.dot(U_r, H_r.T) / (np.dot(WH_r, H_r.T) + self.mu_W)
        return W_update

    def adv_W_update(self, U_r, U_z, W, H_r, H_z):
        """
        Update rule for adversarial NMF
        """

        WH_r = np.matmul(W,H_r)
        WH_z = np.matmul(W,H_z)
        invN_r = 1.0/U_r.shape[1]
        invN_z = 1.0/U_z.shape[1]
        W_update = (np.dot(WH_z, H_z.T) * invN_z + np.dot(U_r,H_r.T) * invN_r) \
        /(np.dot(WH_r, H_r.T)*invN_r + np.dot(U_z,H_z.T)*invN_z + self.mu_W)
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


    def initializeWH(self, U_r = None, U_z = None, U_sup = None, V_sup = None, prob = "std"):
        """
        Initialize W and H for fitting
        Input:

            prob: string. Problem type. 'std' for standard NMF, 'adv' for adversarial, 'sup' for supervised
            type: string. Type of initialization. 'rand' for random, 'exem' for exemplar based, which means we also have to pass U_r
            
            TO DO:
            For exem: Initial vectors should be sampled from both U_r and U_sup
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

    def std_loss(self,U_r, WH_r):
        return 1/U_r.shape[1] * np.linalg.norm(U_r - WH_r, 'fro')**2
    
    def adv_loss(self,U_r,U_z,WH_r,WH_z):
        return 1/U_r.shape[1] * np.linalg.norm(U_r - WH_r, 'fro')**2 - self.tau_A/U_z.shape[1] * np.linalg.norm(U_z - WH_z, 'fro')**2

    def fit_std(self, U_r, conv = False):
        """
        Fits standard NMF by solving

        min_{W \ge 0} 1/N \|U - WH(U)\|_F^2
        where H(U) = arg min_{H \ge 0} \|U - WH\|_F^2

        Fitting is done using a mini-batch multiplicative algorithm that is initialization sensitive
        """

        self.M = U_r.shape[0]
        self.N_r = U_r.shape[1]

        U_r_ = np.copy(U_r)
        
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
        if conv or self.verbose:
            loss_func = self.std_loss
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

            # Update H
            self.H_r = self.H_r * H_update(U_r_,self.W,self.H_r)

            if self.verbose or conv:
                WH_r = np.dot(self.W,self.H_r)
                loss_std[i+1] = loss_func(U_r_,WH_r)
                if self.verbose:
                    print(f"Epoch: {i+1}, Loss: {loss_std[i+1]}")
        if conv:
            return loss_std

    def fit_adv(self, U_r, U_z, conv = False):
        """
        Fits adversarial NMF by solving

        min_{W \ge 0} 1/N \|U - WH(U)\|_F^2 - \tau/\hat{N} \|U - WH(\hat{U})\|_F^2 
        where H(U) = arg min_{H \ge 0} \|U - WH\|_F^2

        Here U is true data and \hat{U} is adversarial data.

        Fitting is done using a mini-batch multiplicative algorithm that is initialization sensitive
        """
        self.M = U_r.shape[0]
        self.N_r = U_r.shape[1]

        assert self.M == U_z.shape[0], f"U_r has first axis {self.M} and U_z has first axis {U_z.shape[0]} which does not match."

        self.N_z = U_z.shape[1]
        U_z_ = np.sqrt(self.tau_A) * U_z

        U_r_ = np.copy(U_r)

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
            self.initializeWH(U_r = U_r, U_z = U_z_, prob = "adv")

        # List of ids that will be shuffled
        ids_r = np.arange(0,self.N_r)
        ids_z = np.arange(0,self.N_z)

        # Set update
        W_update = self.adv_W_update
        H_update = self.H_update
        
        if conv or self.verbose:
            loss_func = self.adv_loss
            loss_adv = np.zeros((self.epochs + 1))
            WH_r = np.dot(self.W,self.H_r)
            WH_z = np.dot(self.W,self.H_z)
            loss_adv[0] = loss_func(U_r,U_z_, WH_r, WH_z)
            

        for i in range(self.epochs):
            # Shuffle ids, U and H
            np.random.shuffle(ids_r)
            np.random.shuffle(ids_z)
            U_r_ = U_r_[:,ids_r]
            self.H_r = self.H_r[:,ids_r]
            U_z_ = U_z_[:,ids_z]
            self.H_z = self.H_z[:,ids_z]  

            # Split into batches
            U_r_b = np.split(U_r_, N_batches_r, axis = 1)
            H_r_b = np.split(self.H_r, N_batches_r, axis = 1)
            U_z_b = np.split(U_z_, N_batches_z, axis = 1)
            H_z_b = np.split(self.H_z, N_batches_z, axis = 1) 

            for b in range(N_batches):

                # Update W for each batch
                self.W = self.W * W_update(U_r_b[b%N_batches_r], U_z_b[b%N_batches_z], self.W, H_r_b[b%N_batches_r], H_z_b[b%N_batches_z])

                
            WtW = np.dot(self.W.T, self.W)
            self.H_r = self.H_r * H_update(U_r_,self.W, self.H_r, WtW = WtW) 
            self.H_z = self.H_z * H_update(U_z_,self.W, self.H_z, WtW = WtW)

            if self.verbose or conv:
                WH_r = np.dot(self.W,self.H_r)
                WH_z = np.dot(self.W,self.H_z)
                loss_adv[i+1] = loss_func(U_r_,U_z_, WH_r, WH_z)
                if self.verbose:
                    print(f"Epoch: {i+1}, Loss: {loss_adv[i+1]}")
        if conv:
            return loss_adv

    def fit_sup(self,U_sup,V_sup,conv = False):
        """
        Solves:

        \min_{W \ge 0} \sum_i^{S} \|U_i - W_i H_i\|_F^2
        s.t H = \argmin_{H} \|V - WH\|_F^2,
        where W = [W_1,...W_S] and H = [H_1,... W_S] are concatenated versions of W and H

        Right now function takes in U_r as a list of (M,N) arrays, it should handle a (M,S,N) array too. 
        input:
           U_r: (M,S,N) array
        """
        assert self.ds is not None and self.d is not None, "Supervised fitting needs both d and ds"
        assert len(U_sup) == len(self.ds), "U_sup and self.ds shapes do not match"
        #self.Ms = np.zeros(len(U_r), dtype = int)

        U_sup_ = []
        for i in range(len(U_sup)):
            U_sup_.append(np.copy(U_sup[i])) 
        V_sup_ = np.copy(V_sup)

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
            V_sup_ = V_sup_[:,ids]
            self.H_sup = self.H_sup[:,ids]

            # Iterate over each source
            for j in range(self.S):

                U_sup_[j] = U_sup_[j][:,ids]

                # Split into batches
                U_sup_b = np.split(U_sup_[j], self.N_batches, axis = 1)
                H_sup_b = np.split(self.H_sup[class_ids[j],:], self.N_batches, axis = 1) 

                for b in range(self.N_batches):

                    # For each batch, calculate update
                    W_up = W_update(U_sup_b[b], self.W[:,class_ids[j]], H_sup_b[b])

                    # Update W
                    self.W[:,class_ids[j]] = self.W[:,class_ids[j]] * W_up

            self.H_sup = self.H_sup * H_update(V_sup_,self.W,self.H_sup)
            
            if conv:
                for j in range(self.S):
                    WH_sup = np.dot(self.W[:,class_ids[j]], self.H_sup[class_ids[j],:])
                    loss_sup[j,i+1] = loss_func(U_sup_[j], WH_sup)
        if conv:
            return loss_sup

    def fit_exem(self, U_r = None):
        """
        Fits Exemplar-based NMF. 
        """

        self.N_r = U_r.shape[1]

        # Fitting of exemplar-based NMF is handled by the initialization function
        self.initializeWH(U_r = U_r)

    
    def fit_full(self, U_r, U_z, U_sup, V_sup, conv = False):
        """
        Fit function that can handle fitting for fitting FNMF which includes weak supervision data, adversarial data
        and strong supervision data. Thus it can be used both to fit a single (A)NMF as well as S FNMFs together

        TO DO:
        - CURRENT IMPLEMENTATION DOES NOT HANDLE UNBALANCED DATA

        """
        # Safety copies
        V_sup_ = np.sqrt(self.tau_S) * V_sup
        U_r_ = []
        U_sup_ = []
        U_z_ = []
        for i in range(len(U_sup)):
            U_r_.append(np.sqrt(1 - self.tau_S) * U_r[i])
            U_sup_.append(np.sqrt(self.tau_S) * U_sup[i])
            U_z_.append(np.sqrt((1 - self.tau_S) * self.tau_A) * U_z[i])
               
        # Store sizes
        self.N_r = U_r_[0].shape[1]
        self.N_z = U_z_[0].shape[1]
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
        
        print(N_batches, N_batches_r,N_batches_z, N_batches_sup)
        class_ids = []
        for i in range(len(self.ds)):
            class_ids.append(np.arange(sum(self.ds[:i]), sum(self.ds[:i+1])))
        

        ids_r = np.arange(0,self.N_r)
        ids_z = np.arange(0,self.N_z)
        ids_sup = np.arange(0,self.N_sup)

        if conv:
            loss_func = lambda U_r, U_z, U_sup, WH_r, WH_z, WH_sup : self.adv_loss(U_r,U_z, WH_r, WH_z) + self.std_loss(U_sup, WH_sup)
            loss_full = np.zeros((self.S, self.epochs + 1))
            for j in range(self.S):
                WH_r = np.dot(self.W[:,class_ids[j]], self.H_r[class_ids[j],:]) 
                WH_z = np.dot(self.W[:,class_ids[j]], self.H_z[class_ids[j],:]) 
                WH_sup = np.dot(self.W[:,class_ids[j]], self.H_sup[class_ids[j],:])
                loss_full[j,0] = loss_func(U_r_[j], U_z[j], U_sup_[j], WH_r, WH_z, WH_sup)


        # Start main loop
        for i in range(self.epochs):
            # Shuffle data

            np.random.shuffle(ids_r)
            np.random.shuffle(ids_z)
            np.random.shuffle(ids_sup)

            V_sup_ = V_sup_[:,ids_sup]
            self.H_sup = self.H_sup[:,ids_sup]
            self.H_r = self.H_r[:,ids_r]
            self.H_z = self.H_z[:,ids_z]  

            # Update W for each source
            for j in range(self.S):


                U_sup_[j] = U_sup_[j][:,ids_sup]
                U_r_[j] = U_r_[j][:,ids_r]
                U_z[j] = U_z[j][:,ids_z]

                # Split data into batches
                U_sup_b = np.split(U_sup_[j], N_batches_sup, axis = 1)
                H_sup_b = np.split(self.H_sup[class_ids[j],:], N_batches_sup, axis = 1) 
                U_r_b = np.split(U_r_[j], N_batches_r, axis = 1)
                H_r_b = np.split(self.H_r[class_ids[j],:], N_batches_r, axis = 1)
                U_z_b = np.split(U_z[j], N_batches_z, axis = 1)
                H_z_b = np.split(self.H_z[class_ids[j],:], N_batches_z, axis = 1) 
                
                for b in range(N_batches):
                    # Calculate terms needed for W update
                    top = np.dot(U_r_b[b%N_batches_r], H_r_b[b%N_batches_r].T) * inv_N_r
                    top += np.dot(self.W[:,class_ids[j]], np.dot(H_z_b[b%N_batches_z], H_z_b[b%N_batches_z].T)) * inv_N_z
                    top += np.dot(U_sup_b[b%N_batches_sup], H_sup_b[b%N_batches_sup].T) * inv_N_sup

                    bot = np.dot(self.W[:,class_ids[j]], np.dot(H_r_b[b%N_batches_r], H_r_b[b%N_batches_r].T)) * inv_N_r
                    bot += np.dot(U_z_b[b%N_batches_z], H_z_b[b%N_batches_z].T) * inv_N_z
                    bot += np.dot(self.W[:,class_ids[j]], np.dot(H_sup_b[b%N_batches_sup], H_sup_b[b%N_batches_sup].T)) * inv_N_sup

                    # Calculate W update
                    self.W[:,class_ids[j]] = self.W[:,class_ids[j]] * (top)/(bot + self.mu_W)

                # Update H_r and H_z for each class
                self.H_r[class_ids[j],:] = self.H_r[class_ids[j],:] * self.H_update(U_r_[j],self.W[:,class_ids[j]],self.H_r[class_ids[j],:]) 
                self.H_z[class_ids[j],:] = self.H_z[class_ids[j],:] * self.H_update(U_z[j], self.W[:,class_ids[j]],self.H_z[class_ids[j],:]) 

            # Calculate H_sup
            self.H_sup = self.H_sup * self.H_update(V_sup_,self.W,self.H_sup) 

            if conv:
                for j in range(self.S):
                    WH_r = np.dot(self.W[:,class_ids[j]], self.H_r[class_ids[j],:]) 
                    WH_z = np.dot(self.W[:,class_ids[j]], self.H_z[class_ids[j],:]) 
                    WH_sup = np.dot(self.W[:,class_ids[j]], self.H_sup[class_ids[j],:])
                    loss_full[j,i+1] = loss_func(U_r_[j], U_z[j], U_sup_[j], WH_r, WH_z, WH_sup)
        if conv:
            return loss_full

            
                
    def transform(self, U, WtW = None, WtU = None):
        """
        
        """ 
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
    Class for source separation with NMF

    TODO:
        - Make sure that all relevant arguments are passed to NMF
        - Fit for full NMF
        - Decide how to handle the eval function.
            -> Should it return a metric for each source, or should we do some sort of broadcast?
            -> This function might need to have an axis argument too
    """
    def __init__(self, ds, mu_W = 1e-6, mu_H = 1e-6,
    epochs = 25, warm_start_epochs = 25, test_epochs = 25,
    prob = "std", loss = "square", init = "exem",
    batch_size = None, batch_size_z = None, batch_size_sup = None,
    wiener = True, eps = 1e-10, true_sample = "std",
    tau_A = 0.1, tau_S = 0.1, betas = None, omegas = None,
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
        self.tau_A = tau_A
        self.tau_S = tau_S 
        self.betas = betas
        self.omegas = omegas
        self.true_sample = true_sample

        if NMFs is None:
            self.NMFs = []
            for i,d in enumerate(ds):
                self.NMFs.append(NMF(d, ds = ds, tau_A = tau_A,
                    tau_S = tau_S, loss = loss, epochs = epochs,
                    test_epochs = test_epochs, init = init, prob = prob, true_sample = true_sample,
                    batch_size = batch_size, batch_size_z = batch_size_z, batch_size_sup = batch_size_sup,
                    mu_W = mu_W, mu_H = mu_H, verbose = verbose))
        else:
            self.NMFs = NMFs

        self.NMF_concat = NMF(d = sum(ds), ds = ds, loss = loss, epochs = epochs, test_epochs = test_epochs,
            tau_A = tau_A, tau_S = tau_S, init = init, prob = prob, true_sample = true_sample,
            batch_size = batch_size, batch_size_z = batch_size_z, batch_size_sup = batch_size_sup,
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
        
        H_concat = self.NMF_concat.transform(V)

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
                nmf.fit_std(U_r[i])
            self.to_concat()
        
        elif self.prob == "adv":

            assert U_r is not None, "Adverserial fitting requires U_r, but U_r is None"
            
            U_z = self.create_adversarial(U_r, V = V)

            for i,nmf in enumerate(self.NMFs):
                if self.warm_start_epochs > 0:
                    nmf.prob = "std"
                    nmf.epochs = self.warm_start_epochs
                    nmf.fit_std(U_r[i])
                    nmf.epochs = self.epochs
                    nmf.prob = "adv"
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

            U_Z.append(np.concatenate(U_r_, axis = 1))
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


def NMF_sep_unit_tests():
    sep = NMF_separation(ds = [2]*2, prob = "exem")
    U = [np.array([[1,2,3],[1,0,3],[1,0,3]]),np.array([[3,1,1],[1,0,2],[1,0,1]])]
    V = U[0] + U[1]
    sep.fit(U)
    out = sep.separate(V)
    assert sep.NMF_concat.W.shape == (3,4)
    assert sep.NMFs[0].W.shape == (3,2)
    assert sep.NMFs[1].W.shape == (3,2)
    assert sep.NMFs[0].H_r.shape == (2,3)
    assert sep.NMFs[1].H_r.shape == (2,3)
    assert out.shape == (3,2,3)
    print("Exemplar-based fitting OK")

    sep = NMF_separation(ds = [2]*2, prob = "std", batch_size = 3)
    sep.fit(U)
    out = sep.separate(V)
    assert sep.NMF_concat.W.shape == (3,4)
    assert sep.NMFs[0].W.shape == (3,2)
    assert sep.NMFs[1].W.shape == (3,2)
    assert sep.NMFs[0].H_r.shape == (2,3)
    assert sep.NMFs[1].H_r.shape == (2,3)
    assert out.shape == (3,2,3)
    print("Standard fitting OK")

    sep = NMF_separation(ds = [2]*2, prob = "adv", batch_size = 3, batch_size_z = 3)
    sep.fit(U)
    out = sep.separate(V)
    assert sep.NMF_concat.W.shape == (3,4)
    assert sep.NMFs[0].W.shape == (3,2)
    assert sep.NMFs[1].W.shape == (3,2)
    assert sep.NMFs[0].H_r.shape == (2,3)
    assert sep.NMFs[1].H_r.shape == (2,3)
    assert out.shape == (3,2,3)
    print("Adversarial fitting OK")

    sep = NMF_separation(ds = [2]*2, prob = "sup", batch_size_sup = 3)
    sep.fit(U_sup = U, V_sup = V)
    out = sep.separate(V)
    assert sep.NMF_concat.W.shape == (3,4)
    assert sep.NMFs[0].W.shape == (3,2)
    assert sep.NMFs[1].W.shape == (3,2)
    assert sep.NMF_concat.H_sup.shape == (4,3)
    #assert out.shape == (3,2,3)
    print("Discriminative fitting OK")

    sep = NMF_separation(ds = [2]*2, prob = "full")
    sep.fit(U_r = U, V = V, U_sup = U, V_sup = V)
    out = sep.separate(V)
    assert sep.NMF_concat.W.shape == (3,4)
    assert sep.NMF_concat.H_sup.shape == (4,3)
    assert sep.NMF_concat.H_r.shape == (4,3)
    print("Full fitting OK")











            


