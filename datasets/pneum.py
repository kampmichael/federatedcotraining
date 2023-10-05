from torchvision import transforms, datasets
import numpy as np
from datasets import *
import math
import os
import pickle
import torch


@is_dataset
class Pneum(Dataset):  
    def __init__(self, method, num_clients, num_unlabeled = 0, num_samples_per_client = -1, test_proportion = 0.33,pickle_files=True, path="data/brain_tumor_dataset/"):
        super(Pneum, self).__init__(method, num_clients, num_unlabeled, num_samples_per_client)
        #Paths
        traindir = r'data/pneumonia/data/preprocessed/train/'
        valdir = r'data/pneumonia/data/preprocessed/val/'
        paths = {"X_train":"Pneumonia_tmp_X_train.pckl","y_train":"Pneumonia_tmp_y_train.pckl",
                 "X_test":"Pneumonia_tmp_X_test.pckl","y_test":"Pneumonia_tmp_y_test.pckl"} 
        X_train,y_train,X_test,y_test = None,None,None,None 
        if pickle_files and self.checkTmpFilesExist(paths):
            X_train,y_train,X_test,y_test = pickle.load(open(paths["X_train"],'rb')), pickle.load(open(paths["y_train"],'rb')),pickle.load(open(paths["X_test"],'rb')), pickle.load(open(paths["y_test"],'rb'))         
            print("Succesfully loaded pickled Penumonia dataset.")
        else: 
            # Training data transforms
            train_transforms = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),])
            val_transforms = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),])

            trainloader = datasets.ImageFolder(traindir, transform=train_transforms)
            testloader = datasets.ImageFolder(valdir, transform=val_transforms)

            X_train = np.array([sample[0].numpy() for sample in trainloader])
            y_train = np.array([sample[1] for sample in trainloader]).astype(np.int_)
            X_test = np.array([sample[0].numpy() for sample in testloader])
            y_test = np.array([sample[1] for sample in testloader]).astype(np.int_)

            pickle.dump(X_train, open(paths["X_train"], 'wb'))
            pickle.dump(y_train, open(paths["y_train"], 'wb'))
            pickle.dump(X_test, open(paths["X_test"], 'wb'))
            pickle.dump(y_test, open(paths["y_test"], 'wb'))

            print("Dataset Pneumonia loaded successfully and pickled.")

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

        
        print("Loaded dataset: xtrain ",self.Xtrain.shape, " ytrain ",self.ytrain.shape," Xtest ", self.Xtest.shape," ytest ", self.ytest.shape, " xunlabeled ",self.Xunlabeled.shape, " yunlabeled ",self.yunlabeled.shape," localidx ", len(self.local_idxs), " localbatchpos ", self.local_batch_pos.shape)


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
    
    def checkTmpFilesExist(self, paths):
        for key in paths:        
            if not os.path.exists(paths[key]):
                print(key, " does not exist: ", paths[key])
                return False
        return True

