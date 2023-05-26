import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils
from datasets import * 
import numpy as np
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from fedlab.utils.dataset.partition import FMNISTPartitioner
from fedlab.utils.functional import partition_report




@is_dataset
class Noniid_FashionMNIST(Dataset):
    def __init__(self, method, num_clients, num_unlabeled = 0, num_samples_per_client = -1, test_proportion = 0.33):
        super(Noniid_FashionMNIST, self).__init__(method, num_clients, num_unlabeled, num_samples_per_client)

        num_classes = 10
        seed = 2021
        hist_color = '#4169E1'

        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False,download=True, transform=transforms.ToTensor())
                        
        X_train, y_train = trainset.data, np.array(trainset.targets)  
        X_test, y_test   = testset.data, np.array(testset.targets)    
                
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

        #self.train_idxs=self.train_idxs[:200*self.num_clients]
        
        # Generating a Non iid part 
        noniid_labeldir_part1 = FMNISTPartitioner(self.ytrain[self.train_idxs][:9000], num_clients=self.num_clients, partition="noniid-labeldir",  dir_alpha=2, seed=seed)
        # generate partition report
        csv_file1 = "fmnist_noniid_labeldir_part1_clients_5.csv"
        partition_report(self.ytrain[self.train_idxs[:9000]], noniid_labeldir_part1.client_dict, class_num=num_classes, verbose=False, file=csv_file1)
        noniid_labeldir_part_df1 = pd.read_csv(csv_file1,header=1)
        noniid_labeldir_part_df1 = noniid_labeldir_part_df1.set_index('client')
        col_names = [f"class{i}" for i in range(num_classes)]
        for col in col_names:
            noniid_labeldir_part_df1[col] = (noniid_labeldir_part_df1[col] * noniid_labeldir_part_df1['Amount']).astype(int)
        
        noniid_labeldir_part_df1[col_names].plot.barh(stacked=True)  
        # plt.tight_layout()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('sample num')
        plt.savefig(f"fmnist_noniid_labeldir_part1_clients_5.png", dpi=400, bbox_inches = 'tight')


        # Generating an iid part  
        noniid_labeldir_part2 = FMNISTPartitioner(self.ytrain[self.train_idxs][9000:], num_clients=self.num_clients, partition="noniid-labeldir",  dir_alpha=100, seed=seed)
        # generate partition report
        csv_file2 = "fmnist_noniid_labeldir_part2_clients_5.csv"
        partition_report(self.ytrain[self.train_idxs[9000:]], noniid_labeldir_part2.client_dict, class_num=num_classes, verbose=False, file=csv_file2)
        noniid_labeldir_part_df2 = pd.read_csv(csv_file2,header=1)
        noniid_labeldir_part_df2 = noniid_labeldir_part_df2.set_index('client')
        col_names = [f"class{i}" for i in range(num_classes)]
        for col in col_names:
            noniid_labeldir_part_df2[col] = (noniid_labeldir_part_df2[col] * noniid_labeldir_part_df2['Amount']).astype(int)
        
        noniid_labeldir_part_df2[col_names].plot.barh(stacked=True)  
        # plt.tight_layout()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('sample num')
        plt.savefig(f"fmnist_noniid_labeldir_part2_clients_5.png", dpi=400, bbox_inches = 'tight')

        self.local_idxs=[]
        for i in range(self.num_clients):
            self.local_idxs.append(np.append(self.train_idxs[:9000][noniid_labeldir_part1.client_dict[i]],self.train_idxs[9000:][noniid_labeldir_part2.client_dict[i]]))


      
        #self.local_idxs = np.array_split(self.train_idxs, self.num_clients)
        self.local_batch_pos = np.zeros(len(self.local_idxs)).astype(int) 

        print("Loaded dataset: xtrain ",self.Xtrain.shape, " ytrain ",self.ytrain.shape," Xtest ", self.Xtest.shape," ytest ", self.ytest.shape, " xunlabeled ",self.Xunlabeled.shape, " yunlabeled ",self.yunlabeled.shape," localidx ", len(self.local_idxs), " localbatchpos ", self.local_batch_pos.shape)

        
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

