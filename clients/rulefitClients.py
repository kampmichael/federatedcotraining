from rulefit import RuleFit #pip install git+https://github.com/christophM/rulefit.git
from clients import *

@is_client 
class RuleFitClassifier(Client):
    def __init__(self):
        super(RuleFitClassifier, self).__init__()
        self.name = "RuleFitClassifier"
        self.ruleFitKWARGs = {"tree_size":4, "sample_fract":'default', "max_rules":200,
             "memory_par":0.01, "rfmode":'classify', "lin_trim_quantile":0.025,
             "lin_standardise":True, "exp_rand_tree_size":True, "random_state":1,"model_type":"r1"}
        self.initRulefit(**self.ruleFitKWARGs)
#"max_iter":1000
    def train(self, X,y):
        #data = pd.read_csv("data/Breast_cancer.csv")
        #data = data.drop(["id", "diagnosis", "Unnamed: 32"], axis=1) 
        #features = data.columns 
        self.initRulefit(**self.ruleFitKWARGs)
        self.model.fit(X,y)
        
    def predict(self, X):
        return self.model.predict(X).round()
    
    def initRulefit(self, tree_size=4, sample_fract='default', max_rules=200, memory_par=0.01, rfmode='classify', lin_trim_quantile=0.025,
             lin_standardise=True, exp_rand_tree_size=True, random_state=1,model_type="r1"):
        self.model = RuleFit(tree_size=tree_size, sample_fract=sample_fract, max_rules=max_rules,
             memory_par=memory_par, rfmode=rfmode, lin_trim_quantile=lin_trim_quantile,
             lin_standardise=lin_standardise, exp_rand_tree_size=exp_rand_tree_size, random_state=random_state,model_type=model_type)
        