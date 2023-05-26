from parameters import *
import numpy as np

def is_client(original_class):
    original_class.baseclass = False
    return original_class

class Client:
    baseclass = True
    
    def __init__(self): #todo: add kwargs so that all clients can have individual parameters for initialization
        self.name = "client (base class)"
        
    def train(self, X,y):
        pass
        
    def predict(self, X):
        pass
    
    def predict_soft(self, X):
        import torch
        if self._mode == 'gpu':
            exampleTensor = torch.tensor(X, dtype=torch.float32, device=self._device)
        else:
            exampleTensor = torch.FloatTensor(X)
        output = self._core(exampleTensor).data.cpu().numpy()
        soft_decisions = torch.softmax(torch.tensor(output), dim=1).numpy()
        return soft_decisions
        
    def getParameters(self): #needs to be implemented for FL
        pass
        
    def setParameters(self): #needs to be implemented for FL
        pass
        
    def __str__(self):
        return self.name