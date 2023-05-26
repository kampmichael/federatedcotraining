import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils
from datasets import * 
import numpy as np
import os
import math

n_holes = 1
length = 16

@is_dataset
class FashionMNIST(Dataset):
    def __init__(self, method, num_clients, num_unlabeled = 0, num_samples_per_client = -1, test_proportion = 0.33):
        super(FashionMNIST, self).__init__(method, num_clients, num_unlabeled, num_samples_per_client)

        #transformer = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x.repeat(3, 1, 1)),transforms.Resize(32),transforms.Normalize((0.5,), (0.5,))],Cutout(n_holes=1, length=16))
        self.train_transform = transforms.Compose([
            #transforms.AutoAugment(),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            #transforms.Resize(32),
            #Cutout(n_holes=n_holes, length=length),
            #transforms.Normalize((0.5,), (0.5,))
            ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            #transforms.Resize(32),
            #transforms.Normalize((0.5,), (0.5,))
            ])

        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,download=True, transform=self.train_transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False,download=True, transform=self.test_transform)

                                       
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
      
        self.local_idxs = np.array_split(self.train_idxs, self.num_clients)
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

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img