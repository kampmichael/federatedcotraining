import numpy as np
import pickle, os
from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime
import wandb



def log_errors(clients, dataset, error_measure, error_log, t):
    train_errors, test_errors  = [], []
    i=0
    
    for client in clients:
        #if type(dataset).__name__=="SVHN":
         #   ytrainpred = client.predict(dataset.data[dataset.local_idxs[i]])
          #  train_errors.append(error_measure(dataset.target[dataset.local_idxs[i]], ytrainpred))
        #else:
        ytrainpred = client.predict(dataset.Xtrain[dataset.local_idxs[i]])
        train_errors.append(error_measure(dataset.ytrain[dataset.local_idxs[i]], ytrainpred))
        ypred = client.predict(dataset.Xtest)
        test_errors.append(error_measure(dataset.ytest, ypred))
        i=i+1
         
    """
        #if len(dataset.Xtrain)==5286: # Pneum dataset. This is a shit way to do it :D
        train_error_round=[]
        chunksX=np.array_split(dataset.Xtrain, 25)
        chunksY=np.array_split(dataset.ytrain, 25)
        for i in range(len(chunksX)):
            ytrainpred= client.predict(chunksX[i])
            train_error_round.append(error_measure(chunksY[i], ytrainpred))
        train_errors.append(np.mean(np.array(train_error_round)))
        ypred = client.predict(dataset.Xtest)
        test_errors.append(error_measure(dataset.ytest, ypred))
    """
       
        #else:                                                                                                             
         #   ytrainpred = client.predict(dataset.Xtrain)
          #  train_errors.append(error_measure(dataset.ytrain, ytrainpred))
           # ypred = client.predict(dataset.Xtest)
            #test_errors.append(error_measure(dataset.ytest, ypred))
    num_clients = min(len(clients), 200)
    error_log[t] = {"train":train_errors,"test":test_errors}
    log_dict = {"test_acc": np.mean(test_errors),"train_acc": np.mean(train_errors),"round": t+1}
    for i in range(num_clients):
        log_dict[f"c{i+1}_test_acc"] = test_errors[i]
        if num_clients > 1:
            log_dict["Standard Deviation"] = np.std(test_errors)
    wandb.log(log_dict)

    '''
    error_log[t] = {"train":train_errors,"test":test_errors}
    if len(clients)==1:
        wandb.log({"test_acc" :np.mean(np.array(test_errors)),"train_acc" :np.mean(np.array(train_errors)),"round ": (t+1)})
    if len(clients)==5:
        wandb.log({"c1_test_acc": np.array(test_errors)[0],"c2_test_acc": np.array(test_errors)[1],"c3_test_acc": np.array(test_errors)[2],"c4_test_acc": np.array(test_errors)[3],"c5_test_acc": np.array(test_errors)[4],
               "test_acc" :np.mean(np.array(test_errors)),"train_acc" :np.mean(np.array(train_errors)),"round ": (t+1), "Standard Deviation": np.std(np.array(test_errors))})
    if len(clients)==10:
         wandb.log({"c1_test_acc": np.array(test_errors)[0],"c2_test_acc": np.array(test_errors)[1],"c3_test_acc": np.array(test_errors)[2],"c4_test_acc": np.array(test_errors)[3],"c5_test_acc": np.array(test_errors)[4],
                    "c6_test_acc": np.array(test_errors)[5],"c7_test_acc": np.array(test_errors)[6],"c8_test_acc": np.array(test_errors)[7],"c9_test_acc": np.array(test_errors)[8],"c10_test_acc": np.array(test_errors)[9],
                    "test_acc" :np.mean(np.array(test_errors)),"train_acc" :np.mean(np.array(train_errors)),"round ": (t+1), "Standard Deviation": np.std(np.array(test_errors))})
    '''
    print("Round ",(t+1),": train ",np.mean(np.array(train_errors))," , test ",np.mean(np.array(test_errors)))


def log_models(clients, t, exp_path):
    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)
    if "torch" in str(clients[0].__class__) or "Torch" in str(clients[0].__class__):    
        for i in range(len(clients)):
            import torch
            torch.save(clients[i]._core.state_dict(), exp_path+"/model_client"+str(i+1)+"_round"+str(t+1)+".model")
    elif "keras" in str(clients[0].__class__) or "Keras" in str(clients[0].__class__):
        for i in range(len(clients)):
            clients[0]._core.save(exp_path+"/model_client"+str(i+1)+"_round"+str(t+1)+".model")
    
def getErrorMeasure(name):
    if name == 'accuracy':
        return accuracy_score
    elif name == 'AUC':
        return roc_auc_score
    else:
        print("ERROR: error measure ",name," unknown.")
        return None
        
def writeLogfiles(exp_path, comm_log, error_log):
    if not os.path.isdir(exp_path):
        os.mkdir(exp_path)
    pickle.dump(comm_log, open(os.path.join(exp_path, "comm_log.pck"), 'wb'))
    pickle.dump(error_log, open(os.path.join(exp_path, "error_log.pck"), 'wb'))
    
def getExpPath(args):
    PREFIX="results/"
    path = PREFIX + args.method + "_" + args.dataset + "_" + args.client + "_" + args.error + "_cl" + str(args.num_clients) + "_rounds" + str(args.num_rounds)
    if args.method == "FedCT":
        path += "_u"+str(args.num_unlabeled) + "_cons" + args.consensus
    path += "_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    return path

def getAvailableClasses(baseclass):
    classList = getAvailableClassesRec(baseclass)
    return {cls.__name__: cls for cls in classList}

def getAvailableClassesRec(baseclass):
    if len(baseclass.__subclasses__()) == 0:
        return [baseclass]
    else:
        available_classes = []
        for cls in baseclass.__subclasses__():
            available_classes += getAvailableClassesRec(cls)
        return available_classes
    


    