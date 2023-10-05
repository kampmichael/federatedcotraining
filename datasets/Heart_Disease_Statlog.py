from torchvision import transforms, datasets
import numpy as np
from datasets import *
import math
import os
import pickle
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler


@is_dataset
class Heart_Disease_Statlog(Dataset):  
    def __init__(self, method, num_clients, num_unlabeled = 0, num_samples_per_client = -1, test_proportion = 0.33,pickle_files=True):
        super(Heart_Disease_Statlog, self).__init__(method, num_clients, num_unlabeled, num_samples_per_client)
        data = pd.read_csv("data/Heart_disease_statlog.csv")
        X=data.drop(columns='target')
        y=data['target']

        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2,random_state = 42 )

        y_train=y_train.to_numpy()
        y_test=y_test.to_numpy()

        #X_train=X_train[:10]
        #y_train=y_train[:10]



        idxs = np.arange(X_train.shape[0])
        np.random.shuffle(idxs)
        self.train_idxs = idxs
        Xunlabeled, yunlabeled = None, None                  
        if num_unlabeled > 0:                    
            self.unlabeled_idxs = idxs[:num_unlabeled]
            self.train_idxs = idxs[num_unlabeled:]
            Xunlabeled = X_train[self.unlabeled_idxs]
            yunlabeled = y_train[self.unlabeled_idxs]

        
        self.Xtrain = X_train
        self.ytrain = y_train 
        self.Xtest = X_test
        self.ytest = y_test
        self.Xunlabeled = Xunlabeled
        self.yunlabeled = yunlabeled
       
        self.local_idxs = np.array_split(self.train_idxs, self.num_clients)
        self.local_batch_pos = np.zeros(len(self.local_idxs)).astype(int)

        
        print("Loaded dataset: xtrain ",self.Xtrain[self.train_idxs].shape, " ytrain ",self.ytrain[self.train_idxs].shape," Xtest ", self.Xtest.shape," ytest ", self.ytest.shape, " xunlabeled ",self.Xunlabeled.shape, " yunlabeled ",self.yunlabeled.shape," localidx ", len(self.local_idxs), " localbatchpos ", self.local_batch_pos.shape)


        #X_train = torch.cuda.FloatTensor(X_train)
        #y_train = torch.tensor(y_train, device=device)
        #X_test = torch.cuda.FloatTensor(X_test)
        #y_test = torch.tensor(y_test, device=device)

    def saveDatasetIndices(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        np.savetxt(os.path.join(path, "unlabeled_idxs.csv"), self.unlabeled_idxs, delimiter=",")
        for i in range(len(self.local_idxs)):
            client_path = os.path.join(path, "client"+str(i+1)+"_idxs.csv")
            np.savetxt(client_path, self.local_idxs[i], delimiter=",")
        
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


