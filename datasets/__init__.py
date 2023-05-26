def is_dataset(original_class):
    original_class.baseclass = False
    return original_class

class Dataset:
    baseclass = True

    def __init__(self, method, num_clients, num_unlabeled = 0, num_samples_per_client = -1, test_proportion = 0.33):
        self.name = "dataset (base class)"
        self.num_clients = num_clients
        
    def getNextLocalBatch(self, i, batch_size):
        pass
        
    def saveDatasetIndices(self, path):
        pass