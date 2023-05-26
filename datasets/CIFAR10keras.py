import tensorflow as tf
from datasets import *
import numpy as np
import math
import os

@is_dataset
class CIFAR10Keras(Dataset):
    def __init__(self, method, num_clients, num_unlabeled = 0, num_samples_per_client = -1, test_proportion = 0.33):
        super(CIFAR10Keras, self).__init__(method, num_clients, num_unlabeled, num_samples_per_client)
        cifar10 = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        input_shape = (32, 32, 3)

        X_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
        X_train = x_train / 255.0
        X_test  = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)
        X_test  = x_test / 255.0                
        
        idxs = np.arange(X_train.shape[0])
        np.random.shuffle(idxs)
        self.train_idxs = idxs
        Xunlabeled, yunlabeled = None, None                  
        if num_unlabeled > 0:                    
            self.unlabeled_idxs = idxs[:num_unlabeled]
            self.train_idxs = idxs[num_unlabeled:]
            Xunlabeled = X_train[self.unlabeled_idxs]
            yunlabeled = y_train[self.unlabeled_idxs]
        
        y_train     = tf.one_hot(y_train.astype(np.int32), depth=10)
        y_test      = tf.one_hot(y_test.astype(np.int32), depth=10)
        yunlabeled  = tf.one_hot(yunlabeled.astype(np.int32), depth=10)
        
        self.Xtrain = X_train
        self.ytrain = y_train.numpy() 
        self.Xtest = X_test
        self.ytest = y_test.numpy()
        self.Xunlabeled = Xunlabeled
        self.yunlabeled = yunlabeled
       
        self.local_idxs = np.array_split(self.train_idxs, self.num_clients)
        self.local_batch_pos = np.zeros(len(self.local_idxs)).astype(int)
        #print("Loaded dataset: xtrain ",self.Xtrain.shape, " ytrain ",self.ytrain.shape," Xtest ", self.Xtest.shape," ytest ", self.ytest.shape, " xunlabeled ",self.Xunlabeled.shape, " yunlabeled ",self.yunlabeled.shape," localidx ", len(self.local_idxs), " localbatchpos ", self.local_batch_pos.shape)
                
        
    def getNextLocalBatch(self, i, batch_size):
        if batch_size == -1: #full batch
            return self.Xtrain[self.local_idxs[i],:], self.ytrain[self.local_idxs[i]]
            
        chunks = math.floor(self.local_idxs[i].shape[0] / batch_size)
        chunks_idxs = np.array_split(self.local_idxs[i], chunks)
        if self.local_batch_pos[i] >= chunks:
            self.local_batch_pos[i] = 0
        #print("Getting next batch at pos ",self.local_batch_pos[i]," which has indices ",chunks_idxs[self.local_batch_pos[i]])
        Xbatch, ybatch = self.Xtrain[chunks_idxs[self.local_batch_pos[i]],:], self.ytrain[chunks_idxs[self.local_batch_pos[i]]]
        self.local_batch_pos[i] += 1
        #print("Unique labels in batch: ", np.unique(ybatch).shape)
        return tf.convert_to_tensor(Xbatch, dtype=tf.float32), tf.convert_to_tensor(ybatch, dtype=tf.float32), 
    
    def saveDatasetIndices(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        np.savetxt(os.path.join(path, "unlabeled_idxs.csv"), self.unlabeled_idxs, delimiter=",")
        for i in range(len(self.local_idxs)):
            client_path = os.path.join(path, "client"+str(i+1)+"_idxs.csv")
            np.savetxt(client_path, self.local_idxs[i], delimiter=",")