import numpy as np
import math
from sklearn.model_selection import train_test_split
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils
import cv2
import pickle
from datasets import *

@is_dataset
class MRI(Dataset):
    def __init__(self, method, num_clients, num_unlabeled = 0, num_samples_per_client = -1, test_proportion = 0.33, path="data/brain_tumor_dataset/", pickle_files=True, seed=1):        
        rng = np.random.default_rng(seed)
        self.num_clients = num_clients
        paths = {"X":"MRI_tmp_X.pckl","y":"MRI_tmp_y.pckl"}
        X,y = None,None
        if pickle_files and self.checkTmpFilesExist(paths):
            X, y = pickle.load(open(paths["X"],'rb')), pickle.load(open(paths["y"],'rb'))         
            print("Succesfully loaded pickled MRI dataset.")
        else: 
            self.paths = []
            for dirname, _, filenames in os.walk(path):
                for filename in sorted(filenames):
                   self.paths.append(os.path.join(dirname, filename))
            self.size=len(self.paths)
            X, y = self.getDataset()
            pickle.dump(X, open(paths["X"], 'wb'))
            pickle.dump(y, open(paths["y"], 'wb'))
            print("Dataset MRI loaded successfully and pickled.")
            ## shuffle
        

        X_test = X[:53].cpu().numpy()
        y_test = y[:53]
        X_train=X[53:]
        y_train=y[53:]

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
        print("Dataset MRI loaded successfully and pickled.")

        
    def saveDatasetIndices(self, path):
        import tensorflow as tf
        if not os.path.isdir(path):
            os.mkdir(path)
        np.savetxt(os.path.join(path, "unlabeled_idxs.csv"), self.unlabeled_idxs, delimiter=",")
        #np.savetxt(os.path.join(path, "test_idxs.csv"), self.test_idxs, delimiter=",")
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

        
    def __len__(self):
        return self.size
    
    def checkTmpFilesExist(self, paths):
        for key in paths:        
            if not os.path.exists(paths[key]):
                print(key, " does not exist: ", paths[key])
                return False
        return True
    
    def get(self, ix, size, pretrained):
        im = cv2.imread(self.paths[ix])[:,:,::-1]
        im = cv2.resize(im,size)
        im = im / 255.
        im = torch.cuda.FloatTensor(im)
        im = im.permute(2,0,1)
        #if pretrained:
        #    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225])
        #    im = normalize(im)
        label = self.paths[ix].split("/")[-2]
        if label == "no":
            target = 0
        if label == "yes":
            target = 1
        return im, label, target
    
    #def __getitem__(self, ix, device, size=(150,150), pretrained=False):
    #    im, label, target = self.get(ix, device, size, pretrained)
    #    return im.to(device).float(),torch.tensor(int(target)).to(device).long()
        
    def getDataset(self, size=(150,150), pretrained=False):
        imgs = []
        labels = []
        targets = []
        for ix in range(self.size):
            #if ix%10==0:
                #print("fetching image ",ix," of ",self.size)
            im, label, target = self.get(ix, size, pretrained)
            imgs.append(im)
            labels.append(label)
            targets.append(target)
        X = torch.stack(imgs)
        y = torch.tensor(np.array(targets))
        return X, y 
            
        