import xgboost as xgb #conda install -c conda-forge xgboost
from clients import Client

class XGBoostClassifier(Client):
    def __init__(self):
        super(XGBoostClassifier, self).__init__()
        self.name = "XGBoostClassifier"
        self.model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
        
    def train(self, X,y):
        self.model.fit(X,y)
        
    def predict(self, X):
        return self.model.predict(X)
        
class XGBoostRegressor(Client):
    def __init__(self):
        super(XGBoostRegressor, self).__init__()
        self.name = "XGBoostRegressor"
        self.model = xgb.XGBRegressor(objective="reg:linear", random_state=42)
        
    def train(self, X,y):
        self.model.fit(X,y)
        
    def predict(self, X):
        return self.model.predict(X)