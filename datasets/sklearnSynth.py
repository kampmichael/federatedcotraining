import numpy as np
import math
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.datasets._samples_generator import _generate_hypercube
from sklearn.utils import check_random_state, shuffle
import os
from datasets import *

@is_dataset
class SklearnSynthetic(Dataset):
    def getNextLocalBatch(self, i, batch_size):
        if batch_size == -1: #full batch
            return self.Xtrain[self.local_idxs[i],:], self.ytrain[self.local_idxs[i]]
            
        chunks = math.floor(self.local_idxs[i].shape[0] / batch_size)
        chunks_idxs = np.array_split(self.local_idxs[i], chunks)
        if self.local_batch_pos[i] >= chunks:
            self.local_batch_pos[i] = 0
        Xbatch, ybatch = self.Xtrain[chunks_idxs[self.local_batch_pos[i]],:], self.ytrain[chunks_idxs[self.local_batch_pos[i]]]
        self.local_batch_pos[i] += 1
        return Xbatch, ybatch

@is_dataset        
class Blobs2D1000(SklearnSynthetic):
    def __init__(self, method, num_clients, num_unlabeled = 0, num_samples_per_client = -1, test_proportion = 0.33, random_state=42):
        super(Blobs2D1000, self).__init__(method, num_clients, num_unlabeled, num_samples_per_client)
        self.name = "Blobs1000"
        self.X, self.y = make_blobs(1000, 2, centers=10, cluster_std=1.0, center_box=(- 10.0, 10.0), random_state=random_state)
        if method == "centralized" or method == "FL":
            self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y, test_size=test_proportion, random_state=random_state)
        elif method == "AIMHI":
            idxs = np.arange(self.X.shape[0])
            np.random.shuffle(idxs)
            self.Xunlabeled = self.X[idxs[:num_unlabeled],:]
            self.yunlabeled = self.y[idxs[:num_unlabeled]] #well, not really, but yeah...
            self.X = self.X[idxs[:num_unlabeled],:]
            self.y = self.y[idxs[:num_unlabeled]]
            self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y, test_size=test_proportion, random_state=random_state)
        else:
            print("Dataset initialization error: method ",method," not recognized.")
            return
        idxs = np.arange(self.Xtrain.shape[0])
        np.random.shuffle(idxs)
        self.local_idxs = np.array_split(idxs, self.num_clients)
        self.local_batch_pos = np.zeros(len(self.local_idxs))
        
    
@is_dataset        
class Blobs2D1000_binClass(SklearnSynthetic):
    def __init__(self, method, num_clients, num_unlabeled = 0, num_samples_per_client = -1, test_proportion = 0.33, random_state=42):
        super(Blobs2D1000_binClass, self).__init__(method, num_clients, num_unlabeled, num_samples_per_client)
        self.name = "Blobs1000"
        self.X, y_temp = make_blobs(1000, 2, centers=10, cluster_std=1.0, center_box=(- 10.0, 10.0), random_state=random_state)
        self.y = np.array([1.0 if y < 5 else 0.0 for y in y_temp])
        if method == "centralized" or method == "FL":
            self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y, test_size=test_proportion, random_state=random_state)
        elif method == "AIMHI":
            idxs = np.arange(self.X.shape[0])
            np.random.shuffle(idxs)
            self.Xunlabeled = self.X[idxs[:num_unlabeled],:]
            self.yunlabeled = self.y[idxs[:num_unlabeled]] #well, not really, but yeah...
            self.X = self.X[idxs[:num_unlabeled],:]
            self.y = self.y[idxs[:num_unlabeled]]
            self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y, test_size=test_proportion, random_state=random_state)
        else:
            print("Dataset initialization error: method ",method," not recognized.")
            return
        idxs = np.arange(self.Xtrain.shape[0])
        np.random.shuffle(idxs)
        self.local_idxs = np.array_split(idxs, self.num_clients)
        self.local_batch_pos = np.zeros(len(self.local_idxs))
    
@is_dataset               
class Diabetes(SklearnSynthetic):
    def __init__(self, method, num_clients, num_unlabeled = 0, num_samples_per_client = -1, test_proportion = 0.33):
        super(DiabetesDataset, self).__init__(method, num_clients, num_unlabeled, num_samples_per_client)
        self.name = "Diabetes"
        Xy = np.genfromtxt('diabetic_preprocessed.csv', delimiter=',')
        self.X, self.y = Xy[:,:-1], Xy[:,-1]
        if method == "centralized" or method == "FL":
            self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y, test_size=test_proportion, random_state=random_state)
        elif method == "AIMHI":
            idxs = np.arange(self.X.shape[0])
            np.random.shuffle(idxs)
            self.Xunlabeled = self.X[idxs[:num_unlabeled],:]
            self.yunlabeled = self.y[idxs[:num_unlabeled]] #well, not really, but yeah...
            self.X = self.X[idxs[:num_unlabeled],:]
            self.y = self.y[idxs[:num_unlabeled]]
            self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(self.X, self.y, test_size=test_proportion, random_state=random_state)
        else:
            print("Dataset initialization error: method ",method," not recognized.")
            return
        idxs = np.arange(self.Xtrain.shape[0])
        np.random.shuffle(idxs)
        self.local_idxs = np.array_split(idxs, self.num_clients)
        self.local_batch_pos = np.zeros(len(self.local_idxs))