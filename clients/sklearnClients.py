from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from clients import *
from parameters import LinearParameters

@is_client
class DT(Client):
    def __init__(self):
        super(DT, self).__init__()
        self.name = "DecisionTree"
        self.model = DecisionTreeClassifier()
        
    def train(self, X,y):
        self.model.fit(X,y)
        
    def predict(self, X):
        return self.model.predict(X)

@is_client        
class RandomForestClassifier(Client):
    def __init__(self):
        super(RandomForestClassifier, self).__init__()
        self.name = "RandomForestClassifier"
        self.model = RandomForestClassifier()
        
    def train(self, X,y):
        self.model.fit(X,y)
        
    def predict(self, X):
        return self.model.predict(X)

@is_client        
class SklearnLinearModel(Client):
    def getParameters(self):
        wb = np.concatenate((self.model.coef_.flatten(), self.model.intercept_))
        #if isinstance(self.model.intercept_, List): #in principle, the intercept can be a list. But this may break at other points, then.
        #    wb = np.array(self.model.coef_[0].tolist() + self.model.intercept_.tolist())
        return LinearParameters(wb)
        
    def setParameters(self, param : LinearParameters): 
        if not isinstance(param, LinearParameters):
            error_text = "The argument param is not of type" + str(LinearParameters) + "it is of type " + str(type(param))
            self.error(error_text)
            raise ValueError(error_text)
        #TODO: so far, we assume that the intercept is a scalar, but it can be also a 1d-array with len > 1. This would have to be configured somehow...
        w = param.get().tolist()
        b = w[-1]
        del w[-1]
        self.model.coef_ = np.array([w])
        self.model.intercept_ = np.array([b])

@is_client        
class RidgeClassification(SklearnLinearModel):
    def __init__(self, alpha = 1.0):
        super(RidgeClassification, self).__init__()
        self.name = "RidgeClassification"
        self.model = RidgeClassifier(alpha=alpha)
        
    def train(self, X,y):
        self.model.fit(X,y)
        
    def predict(self, X):
        return self.model.predict(X)
        
    
@is_client        
class RidgeRegression(SklearnLinearModel):
    def __init__(self, C = 1.0):
        super(RidgeRegression, self).__init__()
        self.name = "RidgeRegression"
        self.model = Ridge(alpha=alpha)
        
    def train(self, X,y):
        self.model.fit(X,y)
        
    def predict(self, X):
        return self.model.predict(X)

@is_client        
class LogisticRegressionClassifier(SklearnLinearModel):
    def __init__(self, alpha = 1.0):
        super(LogisticRegressionClassifier, self).__init__()
        self.name = "LogisticRegressionClassifier"
        self.model = LogisticRegression(C=C)
        
    def train(self, X,y):
        self.model.fit(X,y)
        
    def predict(self, X):
        return self.model.predict(X)