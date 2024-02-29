import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils
from datasets import * 
import numpy as np
import os
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer

@is_dataset
class IMDB(Dataset):
    def __init__(self, method, num_clients, num_unlabeled = 0, num_samples_per_client = -1, test_proportion = 0.33):
        super(IMDB, self).__init__(method, num_clients, num_unlabeled, num_samples_per_client)
         
        df = pd.read_csv('data/IMDB.csv')
        x = df['text'].tolist()
        y = df['label'].tolist()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.add_special_tokens({'pad_token':'<|endoftext|>'})
        tokenizer.padding_side="left"
        tokenized_text = [tokenizer.encode(text, truncation=True, padding='max_length', max_length=128) for text in x]
        x_train, x_test, y_train, y_test = train_test_split(tokenized_text, y, test_size=0.2, random_state=42)
        X_train = torch.tensor(x_train)
        y_train = torch.tensor(y_train)
        X_test = torch.tensor(x_test)
        y_test = torch.tensor(y_test)

        
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
        self.Xtest = X_test[:150]
        self.ytest = y_test[:150]
        self.Xunlabeled = Xunlabeled
        self.yunlabeled = yunlabeled

        # turn on if you need to fix the training set size (default 10k )
        #self.train_idxs=self.train_idxs[:100]  
        #self.train_idxs=self.train_idxs[:20]
      
        self.local_idxs = np.array_split(self.train_idxs, self.num_clients)
        self.local_batch_pos = np.zeros(len(self.local_idxs)).astype(int) 

        print("Loaded dataset: xtrain ",self.Xtrain.shape, " ytrain ",self.ytrain.shape," Xtest ", self.Xtest.shape," ytest ", self.ytest.shape, " xunlabeled ",self.Xunlabeled.shape, " yunlabeled ",self.yunlabeled.shape," localidx ", len(self.local_idxs), " localbatchpos ", self.local_batch_pos.shape)
        print(";;;;;;;;;;;;;;;;;;;;;;")
        print("Train idxs size is :", self.train_idxs.shape)
    
        print("Local_idxs is : ", len(self.local_idxs))

        print(";;;;;;;;;;;;;;;;;;;;;;")
        print("Xtrain[local_idx[1]]", self.Xtrain[self.local_idxs[1]].shape)

        
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
        
    def saveDatasetIndices(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        np.savetxt(os.path.join(path, "unlabeled_idxs.csv"), self.unlabeled_idxs, delimiter=",")
        for i in range(len(self.local_idxs)):
            client_path = os.path.join(path, "client"+str(i+1)+"_idxs.csv")
            np.savetxt(client_path, self.local_idxs[i], delimiter=",")

