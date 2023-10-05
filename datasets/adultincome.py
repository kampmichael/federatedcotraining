from torchvision import transforms, datasets
import numpy as np
from datasets import *
import math
import os
import pandas as pd
from sklearn.model_selection import train_test_split


@is_dataset
class adultincome(Dataset):  
    def __init__(self, method, num_clients, num_unlabeled = 0, num_samples_per_client = -1, test_proportion = 0.33,pickle_files=True):
        super(adultincome, self).__init__(method, num_clients, num_unlabeled, num_samples_per_client)
        df = pd.read_csv("data/adult.csv",na_values='?')
        df['capital-gain'].replace(99999, np.mean(df['capital-gain'].values), inplace=True)
        df['hours-per-week'].replace(99, np.mean(df['hours-per-week'].values), inplace=True)

        df["gender"] = df["gender"].map({'Female':0, 'Male':1})

        df["race"] = df["race"].map({'White':0, 'Black':1, 'Asian-Pac-Islander':2, 'Amer-Indian-Eskimo':3})

        df["marital-status"] = df["marital-status"].map({'Widowed':0, 'Divorced':1, 'Separated':2,'Never-married':3, 
                                                 'Married-civ-spouse':4, 'Married-spouse-absent':5, 'Married-AF-spouse':6})

        df["relationship"] = df["relationship"].map({'Not-in-family':0, 'Unmarried':0, 'Own-child':0, 'Other-relative':0, 
                                             'Husband':1, 'Wife':1})

        df['workclass'] = df['workclass'].map({'?':0, 'Private':1, 'State-gov':2, 'Federal-gov':3, 
                                       'Self-emp-not-inc':4, 'Self-emp-inc': 5, 'Local-gov': 6,
                                       'Without-pay':7, 'Never-worked':8})

        df["income"] = df["income"].map({'<=50K':0, '>50K': 1})
        df.fillna(df.mode().iloc[0], inplace=True)
        df['occupation'] = df['occupation'].apply(lambda x: x.replace('?', 'Unknown'))
        occupation_df = pd.get_dummies(df["occupation"])
        df = pd.concat([df, occupation_df], axis=1)
        df.drop(["occupation"], axis=1, inplace=True)
        df['native-country'] = df['native-country'].apply(lambda x: 1 if x.strip() == "United-States" else 0)
        df[['education', 'educational-num']].groupby(['education'], as_index=False).mean().sort_values(by='educational-num', ascending=False)
        df.drop(["education"], axis=1, inplace=True)

        X = df.drop(["income"], axis=1)
        y = df["income"]


        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0, stratify=y)

        X_train=X_train.to_numpy()
        y_train=y_train.to_numpy()
        X_test=X_test.to_numpy()
        y_test=y_test.to_numpy()



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


