import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils
from datasets import * 
import numpy as np
import os
import math
from torch.utils.data import random_split

n_holes = 1
length = 16

@is_dataset
class SVHN(Dataset):
    def __init__(self, method, num_clients, num_unlabeled = 0, num_samples_per_client = -1, test_proportion = 0.33):
        super(SVHN, self).__init__(method, num_clients, num_unlabeled, num_samples_per_client)

        
        #transformer = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x.repeat(3, 1, 1)),transforms.Resize(32),transforms.Normalize((0.5,), (0.5,))],Cutout(n_holes=1, length=16))
        '''
        self.train_transform = transforms.Compose([
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.SVHN),
            transforms.ToTensor(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            Cutout(n_holes=n_holes, length=length),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        '''
        

        trainset = torchvision.datasets.SVHN(root='data', split='train', download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.SVHN(root='data', download=True, split='test', transform=transforms.ToTensor())

        X_train, y_train = trainset.data, np.array(trainset.labels)  
        X_test, y_test   = testset.data, np.array(testset.labels)


        #data, target = dataset.data, np.array(dataset.labels) 

        #test_size = 12000
        #train_size = len(dataset) - test_size

        #trainset,testset = random_split(dataset, [train_size, test_size])


        #print(len(train_ds), len(val_ds) )   

                                       
        #X_train, y_train = trainset.data, np.array(trainset.targets)  
        #X_test, y_test   = testset.data, np.array(testset.targets)    
                
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