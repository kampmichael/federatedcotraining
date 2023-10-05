import numpy as np
import math
from datasets import *

@is_dataset
class AIMHISynthetic(Dataset):
    def __init__(self, method, num_clients, num_unlabeled = 0, num_samples_per_client = -1, test_proportion = 0.33, n_features = 10, clusters = 8, n_informative = 8, n_redundant = 1, sep = 1.0, classes = 2, flip_proba = 0.0001, seed = None):
        super(AIMHISynthetic, self).__init__(method, num_clients, num_unlabeled, num_samples_per_client)
        self.name = "AIMHI-Synthetic(d="+str(n_features)+")"
        self.num_clients = num_clients
        self.num_unlabeled = num_unlabeled
        self.num_samples_per_client = num_samples_per_client
        self.n_testset = int(num_samples_per_client*num_clients*test_proportion) 
        self.local_batch_pos = np.zeros(num_clients)
        self.client_datasets, self.Xtest, self.ytest, self.Xunlabeled, self.yunlabeled = self.generateSynthDataset(num_samples_per_client, num_unlabeled, self.n_testset, clusters, num_clients, n_features, n_informative, n_redundant, sep, classes, flip_proba, seed)
        self.Xtrain, self.ytrain = np.vstack(tuple([d[0] for d in self.client_datasets])), np.hstack(tuple([d[1] for d in self.client_datasets]))
        
    def getNextLocalBatch(self, i, batch_size):
        X, y = self.client_datasets[i]
        if batch_size == -1: #full batch
            return X,y
        n_samples = y.shape[0]
        chunks = math.floor(n_samples / batch_size)
        if self.local_batch_pos[i] >= chunks:
            self.local_batch_pos[i] = 0
        Xbatch, ybatch = X[self.local_batch_pos[i] * batch_size:(self.local_batch_pos[i]+1) * batch_size,:], y[self.local_batch_pos[i] * batch_size:(self.local_batch_pos[i]+1) * batch_size]
        return Xbatch, ybatch
    
    def generateSynthDataset(self, n_samples_per_client, n_unlabeled, n_test, clusters, n_clients, n_features, n_informative, n_redundant, sep = 1.0, classes = 2, flip_proba = 0.0, seed = None):
        if n_samples_per_client == -1: #indicates that the entire dataset should be equally distributed among clients
            n_samples_per_client = 100 #since there is no dataset size, we choose 100 per client as standard value
        
        majority = n_clients // 2 + 1

        rng = check_random_state(seed)
        centroids = _generate_hypercube(clusters, n_informative, rng).astype(float, copy=False)
        centroids *= 2 * sep
        centroids -= sep
        
        
        class_per_cluster = {}
        for c in range(clusters):
            class_per_cluster[c] = c % classes
        #each cluster is only observed by majority many clients
        start = 0
        stop = majority
        client_clusters = [[] for _ in range(n_clients)]
        for c in range(clusters):
            clients = list(range(start,stop))
            if start > stop:
                clients = list(range(start, n_clients)) + list(range(0,stop))
            for client in clients:
                client_clusters[client].append(c)
            start = stop % n_clients
            stop = (stop + majority) % n_clients
        #generate data for each client
        datasets = []
        for client in range(n_clients):
            samplesPerCluster = [n_samples_per_client // len(client_clusters[client]) if k in client_clusters[client] else 0 for k in range(clusters)]
            overallSamples = sum(samplesPerCluster)
            for i in range(n_samples_per_client - overallSamples):
                samplesPerCluster[client_clusters[client][i % len(client_clusters[client])]] += 1
            X, y = np.zeros((n_samples_per_client, n_features)), np.zeros(n_samples_per_client, dtype=int)            
            X[:, :n_informative] = rng.randn(n_samples_per_client, n_informative)
            stop = 0
            for k in client_clusters[client]:
                centroid = centroids[k]
                start, stop = stop, stop + samplesPerCluster[k]
                y[start:stop] = class_per_cluster[k]
                X_k = X[start:stop, :n_informative] 
                A = 2.0 * rng.rand(n_informative, n_informative) - 1
                X_k[...] = np.dot(X_k, A)  
                X_k += centroid 
            if n_redundant > 0:
                B = 1. * rng.rand(n_informative, n_redundant) - 1
                X[:, n_informative:n_informative + n_redundant] = \
                    np.dot(X[:, :n_informative], B)
            if n_features - n_informative - n_redundant > 0:
                X[:, -(n_features - n_informative - n_redundant):] = rng.randn(n_samples_per_client, (n_features - n_informative - n_redundant))
            flip_mask = rng.rand(n_samples_per_client) < flip_proba
            y[flip_mask] = rng.randint(classes, size=flip_mask.sum())
            X, y = shuffle(X, y, random_state=rng)
            datasets.append([X, y])
        #generate unlabeled data and test data
        Xu, yu = np.zeros((n_unlabeled + n_test, n_features)), np.zeros(n_unlabeled + n_test, dtype=int)
        Xu[:, :n_informative] = rng.randn(n_unlabeled+ n_test, n_informative)
        samplesPerCluster = [(n_unlabeled + n_test) // clusters for k in range(clusters)]
        while sum(samplesPerCluster) < n_unlabeled + n_test:
            samplesPerCluster[rng.randint(0,len(samplesPerCluster))] += 1
        overallSamples = sum(samplesPerCluster)
        stop = 0
        for k in range(clusters):
            centroid = centroids[k]
            start, stop = stop, stop + samplesPerCluster[k]
            yu[start:stop] = class_per_cluster[k]
            X_k = X[start:stop, :n_informative] 
            A = 2.0 * rng.rand(n_informative, n_informative) - 1
            X_k[...] = np.dot(X_k, A)  
            X_k += centroid 
        if n_redundant > 0:
            B = 1. * rng.rand(n_informative, n_redundant) - 1
            Xu[:, n_informative:n_informative + n_redundant] = \
                np.dot(Xu[:, :n_informative], B)
        if n_features - n_informative - n_redundant > 0:
            Xu[:, -(n_features - n_informative - n_redundant):] = rng.randn(n_unlabeled+n_test, (n_features - n_informative - n_redundant))
        flip_mask = rng.rand(n_unlabeled+n_test) < flip_proba
        yu[flip_mask] = rng.randint(classes, size=flip_mask.sum())
        Xu, yu = shuffle(Xu, yu, random_state=rng)
        Xtest, ytest = Xu[:n_test,:], yu[:n_test]
        Xu, yu = Xu[n_test:,], yu[n_test:]
        return datasets, Xtest, ytest, Xu, yu 
