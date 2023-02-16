import keras
import tensorflow as tf
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import scipy as sp

class MNIST:
    def __init__(self,N_train = 5000, N_test = 1313, batch_size = 32, dataset = "mnist"):
        """
        Loads mnist-like datasets and creates a train and test array that is sorted based on class

        input:
            dataset: str, "mnist" for standard MNIST, "fashion" for fashionMNIST
        output:
            x_train, (5000,10,28,28) array. 5000 training samples from the 10 different classes of 28 times 28 images
            x_test, (800,10,28,28) array. 800 test samples from the 10 different classes of 28 times 28 images
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.N_train = N_train
        self.N_test = N_test

        # Get data and rescale
        if dataset == "mnist":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

            # Smallest amount of data from 
            N = 6313

        elif dataset == "fashion":
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

            N = 7000
        else:
            raise ValueError("Unknown dataset!")

        x_ = np.concatenate((x_train,x_test), axis = 0)
        y_ = np.concatenate((y_train,y_test), axis = 0)
        # Make sure the shape is correct

        # Make sure that the type is correct
        x_ = x_.astype('float32')

        # Normalizing 
        x_/= 255

        # Number of data for each of the classes. 
        # One of the classes only has 5421 data for training and 800-something for test
        # In order to have balanced data we discard extra data
        #self.N_train = 5400
        #self.N_test = 820

        # Number of classes
        self.M = 10
        
        ids = np.random.shuffle(np.arange(0,x_.shape[0]))
        x_ = x_[ids]
        y_ = y_[ids]

        # Arrays that store training data sorted after label
        self.x_train = np.zeros((self.N_train, self.M, 28, 28), dtype = 'float32')
        self.x_test = np.zeros((self.N_test, self.M, 28, 28), dtype = 'float32')

        # Sort after label
        for i in range(self.M):
            self.x_train[:,i,:,:] = x_[np.where(y_ == i)][:N_train,:,:] 
            self.x_test[:,i,:,:] = x_[np.where(y_ == i)][N_train:(N_train + N_test),:,:]

    def generate_linear(self,Ms,alpha = 1.0,N_lin = 5000,N_lin_test= 800,seed = 0):
        """
        OUTDATED functionality is foundin generate_supervised
        """
        np.random.seed(seed)
        self.x_train_lin = self.x_train.reshape((self.N_train, self.M,28*28))
        self.x_test_lin = self.x_test.reshape((self.N_test, self.M,28*28))

        self.M_lin = len(Ms)
        self.N_lin = N_lin
        self.N_lin_test = N_lin_test

        # Arrays that will store the datasets
        self.x_lin_train = np.zeros((self.N_lin, 28*28), dtype = 'float32') # Mixed data
        self.y_lin_train = np.zeros((self.N_lin, self.M_lin, 28*28), dtype = 'float32') # Unmixed data
        self.x_lin_test = np.zeros((self.N_lin_test, 28*28), dtype = 'float32') # Mixed data
        self.y_lin_test = np.zeros((self.N_lin_test, self.M_lin, 28*28), dtype = 'float32') # Unmixed data

        # Weights, will be rescaled to sum to 1
        self.c_lin = np.random.dirichlet([alpha] * self.M_lin, (self.N_lin))
        self.c_lin = self.c_lin.astype('float32')
        self.c_lin_test = np.random.dirichlet([alpha] * self.M_lin, (self.N_lin_test))
        self.c_lin_test = self.c_lin_test.astype('float32')

        # Indexes to sample, want to use same samples multiple times
        ids = np.random.choice(self.N_lin - 1,(self.N_lin,self.M_lin), replace = True)
        ids_test = np.random.choice(self.N_lin_test - 1,(self.N_lin_test,self.M_lin), replace = True)

        # Create datasets
        for i in range(self.N_lin):
            for j,m in enumerate(Ms):
                self.y_lin_train[i,j,:] = np.multiply(self.c_lin[i,j], self.x_train_lin[ids[i,j],m,:])
                self.x_lin_train[i,:] += np.multiply(self.c_lin[i,j], self.x_train_lin[ids[i,j],m,:]) 

        for i in range(self.N_lin_test):
            for j,m in enumerate(Ms):
                self.y_lin_test[i,j,:] = np.multiply(self.c_lin_test[i,j], self.x_test_lin[ids_test[i,j], m, :])
                self.x_lin_test[i,:] += np.multiply(self.c_lin_test[i,j], self.x_test_lin[ids_test[i,j],m,:])
        

    def generate_adverserial(self,Ms, type = "deterministic", Ns = None, N_V = 100, weights = None, pytorch = False, seed = None):
        """
        Generates "synthetic supervised" adverserial dataset, which consists a list
        of dataloaders with real data and one dataloader with synthetically generated
        mixed data. We make sure that none of the data used in the synthetically generated
        data also appear in the real data.

        The synthetically mixed data are mixed as
        v = sum_i c_i u_i,
        where c_i are uniformly distributed between some lower and upper bound and then
        rescaled so that sum_i c_i = 1

        Input:
            Ms: list, contains integers between 0 and 9 which denote the different classes of 
                MNIST data. For example, passing 'Ms = [0,1]' will create datasets using
                class 0 and class 1.
            
        Output:
            adv_loaders: list, contains the different dataloaders for each class.
        """
        if seed is not None:
            np.random.seed(seed)

        

        #Sources
        self.M_adv = len(Ms)
        
        # Generate "adverserial" data
        if Ns is None:
            self.Ns_adv = [100]*self.M_adv
        else:
            self.Ns_adv = Ns 
        self.N_adv_V = N_V

        # Set alphas if they are None, default to standard Dirichlet
        if weights is None:
            if type[:3] == "det":
                weights = [1.0/self.M_adv] * self.M_adv
            elif type[:3] == "dir":
                weights = [1.0] * self.M_adv

        # Arrays that will store the datasets
        self.x_r_train = [] #np.zeros((self.N_adv, self.M_adv, 28, 28), dtype = 'float32')
        self.x_v_train = np.zeros((self.N_adv_V, 28, 28), dtype = 'float32')

        #if pytorch:
        #    # List to store the dataloaders (pytorch datasets)
        #    self.adv_loaders = []

        # Need to create the real and adversarial dataset M times
        for k, ms in enumerate(Ms):

            # Clean data
            self.x_r_train.append(self.x_train[:self.Ns_adv[k],ms,:,:])

        if type[:3] == "det":
            self.c_adv = np.tile(weights, (self.N_adv_V,1))
        elif type[:3] == "dir":
            self.c_adv = np.random.dirichlet(weights, (self.N_adv_V))
            self.c_adv = self.c_adv.astype('float32')

        # Indexes to sample
        idp = np.random.choice(self.N_adv_V - 1,(self.N_adv_V,self.M_adv), replace = True)

        for i in range(self.N_adv_V):
            for j,m in enumerate(Ms):
                self.x_v_train[i,:,:] += np.multiply(self.c_adv[i,j], self.x_train[np.max(self.Ns_adv) + idp[i,j],m,:,:])

            # Create dataloader and append it to list
            #if pytorch:
            #    data = torch.utils.data.TensorDataset(torch.Tensor(self.x_r_train[:,k,:,:].reshape((self.N_adv,1,28,28))), torch.Tensor(self.x_p_train[:,k,:,:].reshape(self.N_adv,1,28,28)))
            #    adv_loader = torch.utils.data.DataLoader(dataset = data, batch_size = self.batch_size, shuffle = True, drop_last = True)
            #    self.adv_loaders.append(adv_loader)

    def generate_supervised(self,Ms,type = "deterministic", weights = None,N_sup = 1000, N_sup_test = 200, pytorch = False, seed = None):
        
        if seed is not None:
            np.random.seed(seed)

        self.M_sup = len(Ms)
        self.N_sup = N_sup
        self.N_sup_test = N_sup_test

        if weights is None:
            if type == "deterministic" or type == "det":
                weights = [1.0/self.M_sup] * self.M_sup
            elif type == "dirichlet" or type == "dir":
                weights = [1.0] * self.M_sup

        # Arrays that will store the datasets
        self.x_sup_train = np.zeros((self.N_sup, 1, 28, 28), dtype = 'float32') # Mixed data
        self.y_sup_train = np.zeros((self.N_sup, self.M_sup, 28, 28), dtype = 'float32') # Unmixed data
        self.x_sup_test = np.zeros((self.N_sup_test, 1, 28, 28), dtype = 'float32') # Mixed data
        self.y_sup_test = np.zeros((self.N_sup_test, self.M_sup, 28, 28), dtype = 'float32') # Unmixed data

        if type == "deterministic" or type == "det":
            self.c_sup = np.tile(weights, (self.N_sup,1)) 
            self.c_sup_test = np.tile(weights, (self.N_sup_test,1)) 

        elif type == "dirichlet" or type == "dir":
            self.c_sup = np.random.dirichlet(weights, (self.N_sup))
            self.c_sup_test = np.random.dirichlet(weights, (self.N_sup_test))

        self.c_sup = self.c_sup.astype('float32')
        self.c_sup_test = self.c_sup_test.astype('float32')

        # Indexes to sample, want to use same samples multiple times
        ids = np.random.choice(self.N_train - 1,(self.N_sup,self.M_sup), replace = True)
        ids_test = np.random.choice(self.N_test - 1,(N_sup_test,self.M_sup), replace = True)

        # Create datasets
        for i in range(self.N_sup):
            for j,m in enumerate(Ms):
                self.y_sup_train[i,j,:,:] = np.multiply(self.c_sup[i,j], self.x_train[ids[i,j], m, :, :])
                self.x_sup_train[i,0,:,:] += np.multiply(self.c_sup[i,j], self.x_train[ids[i,j],m,:,:]) 

        for i in range(self.N_sup_test):
            for j,m in enumerate(Ms):
                self.y_sup_test[i,j,:,:] = np.multiply(self.c_sup_test[i,j], self.x_test[ids_test[i,j], m, :, :])
                self.x_sup_test[i,0,:,:] += np.multiply(self.c_sup_test[i,j], self.x_test[ids_test[i,j],m,:,:])
        if pytorch:
            data = torch.utils.data.TensorDataset(torch.Tensor(self.x_sup_train), torch.Tensor(self.y_sup_train))
            self.sup_loader = torch.utils.data.DataLoader(dataset = data, batch_size = self.batch_size, shuffle = True, drop_last = True)
            data_test = torch.utils.data.TensorDataset(torch.Tensor(self.x_sup_test), torch.Tensor(self.y_sup_test))
            self.sup_loader_test = torch.utils.data.DataLoader(dataset = data_test, batch_size = self.batch_size, shuffle = True, drop_last = True)
