
import numpy as np
import scipy as sp
from utils import *
import os
import librosa

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
        import keras
        import tensorflow as tf
        import torch
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
        ids = np.array([np.random.choice(self.N_train, self.N_sup, replace = False)] * self.M_sup)
        ids_test = np.array([np.random.choice(self.N_test, N_sup_test, replace = False)] * self.M_sup)

        # Create datasets
        for i in range(self.N_sup):
            for j,m in enumerate(Ms):
                self.y_sup_train[i,j,:,:] = np.multiply(self.c_sup[i,j], self.x_train[ids[j,i], m, :, :])
                self.x_sup_train[i,0,:,:] += np.multiply(self.c_sup[i,j], self.x_train[ids[j,i],m,:,:]) 

        for i in range(self.N_sup_test):
            for j,m in enumerate(Ms):
                self.y_sup_test[i,j,:,:] = np.multiply(self.c_sup_test[i,j], self.x_test[ids_test[j,i], m, :, :])
                self.x_sup_test[i,0,:,:] += np.multiply(self.c_sup_test[i,j], self.x_test[ids_test[j,i],m,:,:])
        if pytorch:
            data = torch.utils.data.TensorDataset(torch.Tensor(self.x_sup_train), torch.Tensor(self.y_sup_train))
            self.sup_loader = torch.utils.data.DataLoader(dataset = data, batch_size = self.batch_size, shuffle = True, drop_last = True)
            data_test = torch.utils.data.TensorDataset(torch.Tensor(self.x_sup_test), torch.Tensor(self.y_sup_test))
            self.sup_loader_test = torch.utils.data.DataLoader(dataset = data_test, batch_size = self.batch_size, shuffle = True, drop_last = True)

class audio:
    def __init__(self, ids):

        if ids == None:
            ids = ["1673"]
        
        directory = "Audio"

        # Loop through all files in the directory and its subdirectories
        n = 0
        self.speech = []

        total_seconds = 0
        for root, directories, files in os.walk(directory):
            for file in files:
                # Check if the file extension is .flac
                if file.endswith(".flac") and set(file.split('-')) & set(ids):
                    # Use soundfile to read the audio data
                    file_path = os.path.join(root, file)
                    audio, samplerate = librosa.load(file_path, sr = 16000)
                    self.speech.append(audio / np.max(np.abs(audio)))
                    total_seconds += len(audio)/samplerate

        directory = "wham_noise/tt"

        # Loop through all files in the directory and its subdirectories
        self.noise = []
        i = 0
        number_of_data = 1000
        total_seconds = 0
        for root, directories, files in os.walk(directory):
            for file in files:
                # Check if the file extension is .wav
                if file.endswith(".wav"):
                    # Use soundfile to read the audio data
                    file_path = os.path.join(root, file)
                    audio, samplerate = librosa.load(file_path, sr = 16000)
                    self.noise.append(audio / np.max(np.abs(audio)))
                    total_seconds += len(audio)/samplerate
                    i+=1
                if i == number_of_data:
                    break
            if i == number_of_data:
                    break
        
    def generate(self, snr, seed = None):
        
        if seed is not None:
            np.random.seed(seed)

        snr = snr
        snr_linear = 10**(snr/10)

        N_train = len(self.speech)//2
        N_test = len(self.speech)//2

        speech_ = np.random.permutation(self.speech)
        noise_ = np.random.permutation(self.noise)

        self.speech_train = []

        for i in range(N_train):
            self.speech_train.append(speech_[i])

        self.noisy_test = []
        self.speech_test = []
        self.noise_test = []

        As = []
        a_temp = [0.0,0.0]

        i = 0
        for i in range(N_train, N_test + N_train):
            candidates = [b for b in noise_ if len(b) >= len(speech_[i])]
            if candidates:
                try:
                    n = np.random.choice(candidates)
                except:
                    n = candidates
                start = np.random.randint(0, len(n) - len(speech_[i]))
                end = start + len(speech_[i])

                p_clean = calculate_power(speech_[i])
                p_noise = calculate_power(n[start:end])

                a_temp[0] = 1.0
                a_temp[1] = np.sqrt(p_clean/(p_noise * snr_linear))

                a = [a_temp[0]/np.sum(a_temp), a_temp[1]/np.sum(a_temp)]
                #a = [a_temp[0], a_temp[1]]

                As.append(a)

                self.noisy_test.append(a[0] * speech_[i] + a[1] * n[start:end])
                self.speech_test.append(a[0] * speech_[i])
                self.noise_test.append(a[1] * n[start:end])
            else: 
                continue

        return np.sqrt(np.mean(np.square(np.array(As)[:,0]/np.sum(np.square(np.array(As)),axis = 1))))