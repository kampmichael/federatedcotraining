from parameters_pytorch import PyTorchNNParameters
from collections import OrderedDict
import numpy as np
from clients import *

class PyTorchNN(Client):
    def __init__(self, device_type="cuda"):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        super(PyTorchNN, self).__init__()
        device = torch.device(device_type)
        self._core          		= None
        self._flattenReferenceParams    = None
        self._mode			= 'gpu'
        self._device		= device
        self.name = "PyTorchNN base class"
        self.old_params = None

    def setCore(self, network):
        self._core = network

    def setModel(self, param: PyTorchNNParameters, setReference: bool):
        super(PyTorchNN, self).setModel(param, setReference)
        
        if setReference:
            self._flattenReferenceParams = self._flattenParameters(param)

    def setLoss(self, lossFunction):
        import torch.nn as nn
        self._loss = eval("nn." + lossFunction + "()")

    def setUpdateRule(self, updateRule, learningRate, schedule_ep, schedule_changerate, **kwargs):
        import torch.optim as optim
        additional_params = ""
        for k in kwargs:
            additional_params += ", " + k  + "=" + str(kwargs.get(k))
        self._updateRule = eval("optim." + updateRule + "(self._core.parameters(), lr=" + str(learningRate) + additional_params + ")")
        if (schedule_ep is not None):
            self._schedule = optim.lr_scheduler.StepLR(self._updateRule, schedule_ep, gamma=schedule_changerate, last_epoch=-1, verbose=False)
        else:
            self._schedule = None

    def per_train(self, X1, y1,X2,y2,lmbda):
        import torch
        import torch.nn as nn
        if self._core is None:
            self.error("No core is set")
            raise AttributeError("No core is set")
        self.old_params = self.getParameters() #store previous parameters
        self._updateRule.zero_grad()   # zero the gradient buffers
        def prepare_tensors(X, y):
            if isinstance(X, torch.Tensor):
                X = X.cpu().numpy()
            if isinstance(y, torch.Tensor):
                y = y.cpu().numpy()
            
            if self._mode == 'gpu':
                exampleTensor = torch.cuda.FloatTensor(X, device=self._device)
                labelTensor = torch.tensor(y, device=self._device)      
            else:
                exampleTensor = torch.tensor(X)
                if type(self._loss) is nn.MSELoss or type(self._loss) is nn.L1Loss:
                    labelTensor = torch.tensor(y)
                else:
                    labelTensor = torch.LongTensor(y)
            return exampleTensor, labelTensor
        # Prepare tensors for X1, y1 and X2, y2
        exampleTensor1, labelTensor1 = prepare_tensors(X1, y1)
        exampleTensor2, labelTensor2 = prepare_tensors(X2, y2)
        
        # Forward pass
        output1 = self._core(exampleTensor1)
        output2 = self._core(exampleTensor2)
        
        # Calculate individual losses
        loss1 = self._loss(output1, labelTensor1)
        loss2 = self._loss(output2, labelTensor2)
        # Combine the losses
        loss= loss1 + lmbda * loss2

        loss.backward()
        self._updateRule.step()    
        if self._schedule is not None:
            self._schedule.step()
        return [loss.data.cpu().numpy(),output1.data.cpu().numpy(), output2.data.cpu().numpy()]


    def train(self, X, y):
        import torch
        import torch.nn as nn
        if self._core is None:
            self.error("No core is set")
            raise AttributeError("No core is set")
        self.old_params = self.getParameters() #store previous parameters
        self._updateRule.zero_grad()   # zero the gradient buffers
        exampleTensor = X
        labelTensor = y
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()      
        if self._mode == 'gpu':
            exampleTensor = torch.cuda.FloatTensor(X, device=self._device)
            labelTensor = torch.tensor(y, device=self._device)      
        else:
            exampleTensor = torch.tensor(X)
            if type(self._loss) is nn.MSELoss or type(self._loss) is nn.L1Loss:
                labelTensor = torch.tensor(y)
            else:
                labelTensor = torch.LongTensor(y)
        output = self._core(exampleTensor)
        loss = self._loss(output, labelTensor)
        loss.backward()
        self._updateRule.step()    
        if self._schedule is not None:
            self._schedule.step()
        return [loss.data.cpu().numpy(), output.data.cpu().numpy()]
        
    def predict(self, X) -> np.ndarray:
        import torch
        if self._mode == 'gpu':
            exampleTensor = torch.tensor(X, dtype=torch.float32, device=self._device)
        else:
            exampleTensor = torch.FloatTensor(X)
        output = self._core(exampleTensor).data.cpu().numpy()
        return np.argmax(output, 1)

    def setParameters(self, param : PyTorchNNParameters):
        import torch
        if not isinstance(param, PyTorchNNParameters):
            error_text = "The argument param is not of type" + str(PyTorchNNParameters) + "it is of type " + str(type(param))
            raise ValueError(error_text)

        state_dict = OrderedDict()
        for k,v in param.get().items():
            if self._mode == 'gpu':
                if v.shape == ():
                    state_dict[k] = torch.tensor(v, device=self._device)
                else:
                    state_dict[k] = torch.cuda.FloatTensor(v, device=self._device)
            else:
                if v.shape == ():
                    state_dict[k] = torch.tensor(v)
                else:
                    state_dict[k] = torch.FloatTensor(v)
        self._core.load_state_dict(state_dict)

    def getParameters(self) -> PyTorchNNParameters:
        state_dict = OrderedDict()
        for k, v in self._core.state_dict().items():
            state_dict[k] = v.data.cpu().numpy()
        params = PyTorchNNParameters(state_dict)
        return params
        
    def getPreviousParameters(self) -> PyTorchNNParameters:
        return self.old_params

    def _flattenParameters(self, param):
        flatParam = []
        for k,v in param.get().items():
            flatParam += np.ravel(v).tolist()
        return np.asarray(flatParam)

@is_client        
class Resnet18Cifar10(PyTorchNN):  
    def __init__(self, device_type="cuda"):
        from resnet import Cifar10ResNet18
        super(Resnet18Cifar10, self).__init__()
        self.name = "Resnet18Cifar10"
        device = torch.device(device_type)
        optimizer = 'SGD' #should be given via kwargs, or by passing args from experiment
        lr = 0.01
        lr_schedule_ep = None
        lr_change_rate = 0.5
        lossFunction = "CrossEntropyLoss"
        torchnetwork = Cifar10ResNet18() #Cifar10ResNet50()
        torchnetwork = torchnetwork.cuda(device)
        self.setCore(torchnetwork)
        self.setLoss(lossFunction)
        self.setUpdateRule(optimizer, lr, lr_schedule_ep, lr_change_rate)

@is_client        
class PaperPytorchCIFARNet(PyTorchNN):  
    def __init__(self, device_type="cuda"):
        from papernet import Cifar10paperNetPytorch
        import torch
        super(PaperPytorchCIFARNet, self).__init__()
        self.name = "PaperPytorchCIFARNet"
        device = torch.device(device_type)
        optimizer = 'Adam' #should be given via kwargs, or by passing args from experiment
        lr = 0.01
        lr_schedule_ep = None
        lr_change_rate = 0.05
        lossFunction = "CrossEntropyLoss"
        torchnetwork = Cifar10paperNetPytorch() #Cifar10ResNet50()
        torchnetwork = torchnetwork.cuda(device)
        self.setCore(torchnetwork)
        self.setLoss(lossFunction)
        self.setUpdateRule(optimizer, lr, lr_schedule_ep, lr_change_rate)


@is_client
class PaperPytorchMRINet(PyTorchNN):  
    def __init__(self, device_type="cuda"):
        from papernet import MRInet
        import torch
        import torch.nn as nn
        super(PaperPytorchMRINet, self).__init__()
        self.name = "PaperPytorchMRINet"
        device = torch.device(device_type)
        optimizer = 'Adam' #should be given via kwargs, or by passing args from experiment
        lr = 0.001
        lr_schedule_ep = None
        lr_change_rate = 0.005
        lossFunction = "CrossEntropyLoss"
        torchnetwork = MRInet()
        torchnetwork = torchnetwork.cuda(device)
        self.setCore(torchnetwork)
        self.setLoss(lossFunction)
        self.setUpdateRule(optimizer, lr, lr_schedule_ep, lr_change_rate)

@is_client
class PaperPytorchPNEUMNet(PyTorchNN):  
    def __init__(self, device_type="cuda"):
        from papernet import PNEUMnet
        import torch
        import torch.nn as nn
        super(PaperPytorchPNEUMNet, self).__init__()
        self.name = "PaperPytorchPNEUMNet"
        device = torch.device(device_type)
        optimizer = 'Adam' #should be given via kwargs, or by passing args from experiment
        lr = 0.001
        lr_schedule_ep = None
        lr_change_rate = 0.005
        lossFunction = "CrossEntropyLoss"
        torchnetwork = PNEUMnet()
        torchnetwork = torchnetwork.cuda(device)
        self.setCore(torchnetwork)
        self.setLoss(lossFunction)
        self.setUpdateRule(optimizer, lr, lr_schedule_ep, lr_change_rate)

@is_client
class PaperPytorchFashionMNIST(PyTorchNN):  
    def __init__(self, device_type="cuda"):
        from papernet import FashionMNISTnet
        import torch
        import torch.nn as nn
        super(PaperPytorchFashionMNIST, self).__init__()
        self.name = "PaperPytorchFashionMNIST"
        device = torch.device(device_type)
        optimizer = 'Adam' #should be given via kwargs, or by passing args from experiment
        lr = 0.00001
        lr_schedule_ep = None
        lr_change_rate = 0.00005
        lossFunction = "CrossEntropyLoss"
        torchnetwork = FashionMNISTnet()
        pytorch_total_params = sum(p.numel() for p in torchnetwork.parameters())
        #print("number of network paramters is:",pytorch_total_params)
        torchnetwork = torchnetwork.cuda(device)
        self.setCore(torchnetwork)
        self.setLoss(lossFunction)
        self.setUpdateRule(optimizer, lr, lr_schedule_ep, lr_change_rate)

@is_client
class PaperPytorchSVHN(PyTorchNN):  
    def __init__(self, device_type="cuda"):
        from papernet import SVHNnet
        import torch
        import torch.nn as nn
        super(PaperPytorchSVHN, self).__init__()
        self.name = "PaperPytorchSVHN"
        device = torch.device(device_type)
        optimizer = 'Adam' #should be given via kwargs, or by passing args from experiment
        lr = 1e-3
        lr_schedule_ep = None
        lr_change_rate = 0.0005
        lossFunction = "CrossEntropyLoss"
        torchnetwork = SVHNnet()
        torchnetwork = torchnetwork.cuda(device)
        self.setCore(torchnetwork)
        self.setLoss(lossFunction)
        self.setUpdateRule(optimizer, lr, lr_schedule_ep, lr_change_rate)