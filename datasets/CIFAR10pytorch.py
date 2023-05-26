import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils
from datasets import * 
import numpy as np
import os
import math

@is_dataset
class CIFAR10pytorch(Dataset):
    def __init__(self, method, num_clients, num_unlabeled = 0, num_samples_per_client = -1, test_proportion = 0.33):
        super(CIFAR10pytorch, self).__init__(method, num_clients, num_unlabeled, num_samples_per_client)
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)


        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

                                       
        X_train, y_train = trainset.data, np.array(trainset.targets)  #next(iter(trainloader))[0].numpy(), next(iter(trainloader))[1].numpy()
        X_test, y_test   = testset.data, np.array(testset.targets)    #next(iter(testloader))[0].numpy(), next(iter(testloader))[1].numpy()
                
        idxs = np.arange(X_train.shape[0])
        np.random.shuffle(idxs)
        self.train_idxs = idxs
        
        Xunlabeled, yunlabeled = None, None         
        if num_unlabeled > 0:
            self.unlabeled_idxs = idxs[:num_unlabeled]
            self.train_idxs = idxs[num_unlabeled:]
            Xunlabeled = X_train[self.unlabeled_idxs]
            yunlabeled = y_train[self.unlabeled_idxs]         
            Xunlabeled = torch.cuda.FloatTensor(Xunlabeled).permute(0,3,1,2)
        X_train = torch.cuda.FloatTensor(X_train).permute(0,3,1,2)
        X_test = torch.cuda.FloatTensor(X_test).permute(0,3,1,2)
        
        self.Xtrain = X_train.cpu().numpy()
        self.ytrain = y_train 
        self.Xtest = X_test.cpu().numpy()
        self.ytest = y_test
        self.Xunlabeled = Xunlabeled if Xunlabeled is None else Xunlabeled.cpu().numpy()
        self.yunlabeled = yunlabeled     
      
        self.local_idxs = np.array_split(self.train_idxs, self.num_clients)
        self.local_batch_pos = np.zeros(len(self.local_idxs)).astype(int)                
        
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