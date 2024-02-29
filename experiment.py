import torch
torch.cuda.empty_cache()
import os
import sys
import pkgutil
import importlib
import numpy as np
from utils import *
import argparse
from clients import Client
from clients.kerasNN import *
from clients.pytorchNN import *
from clients.sklearnClients import *
from datasets import Dataset
from datasets.CIFAR10keras import *
from datasets.CIFAR10pytorch import *
from datasets.aimhiSynth import *
from datasets.MRI import *
from datasets.sklearnSynth import *
from datasets.pneum import *
from datasets.FashionMNIST import *
from datasets.SVHN import *
from datasets.Noniid_FashionMNIST import *
from datasets.breast_cancer import *
from datasets.winequalityN import *
from datasets.Heart_Disease_Statlog import *
from datasets.adultincome import *
from datasets.mushrooms import *
from datasets.covtype import *
from clients.rulefitClients import *
from clients.xgboostClients import *
from clients import *
from datasets import *
import aggregators
import inspect
from consensus import *
from xor import *
import copy
import random
import tensorflow as tf
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

available_clients = getAvailableClasses(Client)
available_datasets = getAvailableClasses(Dataset)
torch.cuda.empty_cache()

#set the random seeds
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)

#Weight and Bises login
wandb.login(key='999ae259390532c6c283ea338b365a2e9b489d78')

available_aggregators = []
for name, obj in inspect.getmembers(aggregators):
    if inspect.isclass(obj):
        available_aggregators.append(name)
        
#set the parameters
parser = argparse.ArgumentParser(description='AIM-HI Experiment Suite')
parser.add_argument('--method', type=str, default='centralized',
                    choices=['centralized', 'FedCT', 'FL','FL-DP','FedCT-DP','DD','FedCT-diffclass','PATE','DP-PATE'],
                    help='Federated learing method to be used (default: AIMHI)')
parser.add_argument('--num-clients', type=int, default=1,
                    help='Number of clients in federated network (default: 1)')
parser.add_argument('--num-rounds', type=int, default=10,
                    help='Number of rounds of training (default: 10)')
parser.add_argument('--comm-period', type=int, default=1,
                    help='Number of rounds after which clients communicate (default: 1)')                    
parser.add_argument('--train-batch-size', type=int, default=-1,
                    help='Input batch size for training, -1 means full batch (default: -1)')
parser.add_argument('--evaluation-rounds', type=int, default=10,
                    help='Interval for computing error measures (default: 10)')
parser.add_argument('--num-samples-per-client', type=int, default=-1,
                    help='Number of samples each client has available, -1 means the training set is equally divided (default: -1)')
parser.add_argument('--num-unlabeled', type=int, default=0,
                    help='Number of unlabeled samples, has to be smaller than dataset size (default: 0)')

parser.add_argument('--client', type=str, default=list(available_clients.keys())[0], choices = available_clients,
                    help='Client type used for training (default: '+list(available_clients.keys())[0]+')')
#parser.add_argument('--client', type=str, default=list(available_clients.keys())[0],
 #                   help='Client type used for training (default: '+list(available_clients.keys())[0]+')')

parser.add_argument('--dataset', type=str,  default=list(available_datasets.keys())[0], choices = available_datasets,
                    help='dataset type used for training (default: '+list(available_datasets.keys())[0]+')')
parser.add_argument('--aggregator', type=str, default=available_aggregators[1], choices = available_aggregators,
                    help='aggregator type used for FL (default: '+available_aggregators[1]+')')                    
parser.add_argument('--consensus', type=str, default="majority", choices = ['majority','average'],
                    help='Method for finding consensus label on unlabeled data (default: majority)')
parser.add_argument('--error', type=str, default="accuracy", choices = ['accuracy', 'AUC'],
                    help='Error measure (default: accuracy)')
parser.add_argument('--device-type', type=str, default="CUDA", choices = ['CUDA', 'CPU'],
                    help='Interval for computing error measures (default: 10)')
parser.add_argument('--dp-sigma', type=float, default=0.01,
                    help='Sigma of differential privacy mechanism (default: 0.01)')
parser.add_argument('--dp-s', type=float, default=2,
                    help='S of differential privacy mechanism (default: 2)')
parser.add_argument('--lmbda', type=float, default=1,
                    help='lmbda is a regularization parameter for personalized distributed co-training')
parser.add_argument('--distill-period', type=int, default=1,
                    help='distill period of (default: 2)')
args = parser.parse_args()     

cuda = torch.device('cuda:2')
### Initialize ###
client_class = available_clients[args.client]
num_clients = args.num_clients if args.method != "centralized" else 1
dataset = available_datasets[args.dataset](args.method, num_clients, args.num_unlabeled, args.num_samples_per_client)


if args.dataset == "CIFAR10Keras":
    name="CIFAR10_FedCT"
if args.dataset=="MRI":
    name="MRI_FedCT"
if args.dataset=="Pneum":
    name="Pneum_FedCT"
if args.dataset=="FashionMNIST":
    name="FashionMNIST_FedCT"
if args.dataset=="SVHN":
    name="SVHN_FedCT"
if args.num_clients>5:
    name="Scalability_FedCT"
if args.dataset=="Noniid_FashionMNIST":
    name="Noniid_FashionMNIST_FedCT"
if args.dataset=="breast_cancer":
    name="breast_cancer_FedCT"
if args.dataset=="winequalityN":
    name="winequalityN_FedCT"
if args.dataset=="Heart_disease_statlog":
    name="Heart_disease_statlog_FedCT"
if args.dataset=="adultincome":
    name="adultincome"
if args.dataset=="mushrooms":
    name="mushrooms"
if args.dataset=="covtype":
    name="covtype"
#if args.method=="DP-FL" or "FL-DP":
 #   name="FL-DP"
if args.method=="FedCT-DP":
    name="FedCT-DP"
if args.method=="DD":
    name="DD"
if args.method=="Per-FedCT":
    name="Personalized FedCT"
    
wandb.init(

    project=name,
    # track hyperparameters and run metadata with wandb.config
    config={
        "metric": args.error,
        "num_rounds": args.num_rounds,
        "batch_size": args.train_batch_size,
        "numclients": args.num_clients,
        "commperiod":args.comm_period,
        "evaluationrounds":args.evaluation_rounds,
        "numunlabeled":args.num_unlabeled,
        "clients":args.client,
        "dataset":args.dataset,
        "method":args.method,
        #"dp_sigma":args.dp_sigma,
        #"dp_s": args.dp_s,
        "consensus": args.consensus,
        "lmbda":args.lmbda
        
        
    }
)

config = wandb.config





aggregate_class = getattr(aggregators, args.aggregator)
aggregate = aggregate_class()

error_measure = getErrorMeasure(args.error)
exp_path = getExpPath(args)

dataset.saveDatasetIndices(exp_path)

FedCT_EXPORT_STATEDICT = True

comm_log = {}
error_log = {}

### Training ###

if args.method == "centralized": #standard centralized learning
    client = client_class()
    for t in range(args.num_rounds):
        Xtrain, ytrain = dataset.getNextLocalBatch(0, args.train_batch_size)
        #print('Xtrain is', Xtrain,'ytrain',ytrain, 'Xtrain shape is ', Xtrain.shape , 'ytrain shape is', ytrain.shape)
        client.train(Xtrain, ytrain)
        comm_log[t] = []
        if (t + 1) % args.evaluation_rounds == 0:
            log_errors([client], dataset, error_measure, error_log, t)

elif args.method == "FedCT": #label sharing
    clients = []
    for _ in range(args.num_clients):
        client = client_class()
        clients.append(client)
    predictions = None
    for t in range(args.num_rounds):
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        yconsensus = getConsensusLabels(predictions, args.consensus) #we could avoid that with the same if statement as in the concatenation below, but I choose less code over efficiency ATM
        predictions = []
        for i,client in enumerate(clients):
            Xtrain, ytrain = dataset.getNextLocalBatch(i, args.train_batch_size)
            if (t+1) % args.comm_period == 0 and yconsensus is not None:
                Xtrain, ytrain = np.concatenate((Xtrain, dataset.Xunlabeled)), np.concatenate((ytrain, yconsensus))
            if args.dataset=="IMDB":
                client.train_llm(Xtrain, ytrain)
            else:
                client.train(Xtrain, ytrain)
        if (t+1) % args.comm_period == 0:
            print("Comm-round ",(t+1))
            #params = []
            for client in clients:
                if args.dataset=="IMDB":
                    predictions.append(client.predict_llm(dataset.Xunlabeled))
                else:
                    predictions.append(client.predict(dataset.Xunlabeled))              
                #params.append(client.getParameters())
            #agg_params = aggregate(params)
            predictions = np.array(predictions)
            #comm_log[t] = {"params":params.copy(),"agg":agg_params,"clients":params}
            comm_log[t] = {"predictions" : predictions.copy(), "consensus_labels" : getConsensusLabels(predictions, args.consensus)}
        if args.evaluation_rounds == 1 or (t + 1) % args.evaluation_rounds == 0:
            log_errors(clients, dataset, error_measure, error_log, t)
            log_models(clients, t, exp_path)

elif args.method == "FedCT-DP": #label sharing
    clients = []
    for _ in range(args.num_clients):
        client = client_class()
        clients.append(client)
    predictions = None
    for t in range(args.num_rounds):
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        yconsensus = getConsensusLabels(predictions, args.consensus) #we could avoid that with the same if statement as in the concatenation below, but I choose less code over efficiency ATM
        predictions = []
        for i,client in enumerate(clients):
            Xtrain, ytrain = dataset.getNextLocalBatch(i, args.train_batch_size)
            if (t+1) % args.comm_period == 0 and yconsensus is not None:
                Xtrain, ytrain = np.concatenate((Xtrain, dataset.Xunlabeled)), np.concatenate((ytrain, yconsensus))
            client.train(Xtrain, ytrain)
        if (t+1) % args.comm_period == 0:
            print("Comm-round ",(t+1))
            params = []
            for client in clients:

                #print(client.predict(dataset.Xunlabeled).shape)
                prediction_binary=torch.eye(10)[client.predict(dataset.Xunlabeled)].type(torch.int64)
                #prediction_binary=client.predict(dataset.Xunlabeled)
                #print(prediction_binary.shape)
                #print("prediction binray issss",prediction_binary)
                prediction_binary=xor_mechanism(prediction_binary,args.dp_s,prediction_binary)
                print("dps is .......",args.dp_s)
                #prediction_binary=xor_noise(prediction_binary,args.dp_s)
                prediction_=torch.max(torch.tensor(prediction_binary), 1)[1]
                prediction_=prediction_.numpy()
                #print("change is .......", np.count_nonzero(prediction_==client.predict(dataset.Xunlabeled)))
                predictions.append(prediction_)               
                params.append(client.getParameters())

            agg_params = aggregate(params)
            predictions = np.array(predictions)
            #comm_log[t] = {"params":params.copy(),"agg":agg_params,"clients":params}
            comm_log[t] = {"predictions" : predictions.copy(), "consensus_labels" : getConsensusLabels(predictions, args.consensus)}
        if args.evaluation_rounds == 1 or (t + 1) % args.evaluation_rounds == 0:
            log_errors(clients, dataset, error_measure, error_log, t)
            log_models(clients, t, exp_path)

elif args.method == "FedCT-diffclass": #label sharing
    #clients = []
    #print(available_clients.keys())
    client_names = ['XGBoostClassifier', 'sklearnNeuralNetwork', 'RandomforestClassifier', 'RuleFitClassifier']
    clients = [available_clients[client_names[i % len(client_names)]]() for i in range(args.num_clients)]
    print("Clients are", clients)
    #print('num of clinet is ',args.num_clients)
    #breakpoint()
    predictions = None
    for t in range(args.num_rounds):
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        yconsensus = getConsensusLabels(predictions, args.consensus) #we could avoid that with the same if statement as in the concatenation below, but I choose less code over efficiency ATM
        predictions = []
        for i,client in enumerate(clients):
            Xtrain, ytrain = dataset.getNextLocalBatch(i, args.train_batch_size)
            if (t+1) % args.comm_period == 0 and yconsensus is not None:
                Xtrain, ytrain = np.concatenate((Xtrain, dataset.Xunlabeled)), np.concatenate((ytrain, yconsensus))
            client.train(Xtrain, ytrain)
        if (t+1) % args.comm_period == 0:
            print("Comm-round ",(t+1))
            #params = []
            for client in clients:
                predictions.append(client.predict(dataset.Xunlabeled))                
                #params.append(client.getParameters())
            #agg_params = aggregate(params)
            predictions = np.array(predictions)
            #comm_log[t] = {"params":params.copy(),"agg":agg_params,"clients":params}
            comm_log[t] = {"predictions" : predictions.copy(), "consensus_labels" : getConsensusLabels(predictions, args.consensus)}
        if args.evaluation_rounds == 1 or (t + 1) % args.evaluation_rounds == 0:
            log_errors(clients, dataset, error_measure, error_log, t)
            log_models(clients, t, exp_path)


elif args.method == "FL": #model aggregation (e.g., averaging)
    clients = []
    for _ in range(args.num_clients):
        client = client_class()
        clients.append(client)
    for t in range(args.num_rounds):
        for i,client in enumerate(clients):
            Xtrain, ytrain = dataset.getNextLocalBatch(i, args.train_batch_size)
            if args.dataset=="IMDB":
                client.train_llm(Xtrain, ytrain)
            else:
                client.train(Xtrain, ytrain)
        if (t+1) % args.comm_period == 0:
            print("Comm-round ",(t+1))
            params = []
            for client in clients:
                params.append(client.getParameters())
            agg_params = aggregate(params)
            for client in clients:
                client.setParameters(agg_params)
            comm_log[t] = {"params":params.copy(),"agg":agg_params,"clients":params}
        if args.evaluation_rounds == 1 or (t + 1) % args.evaluation_rounds == 0:
            log_errors(clients, dataset, error_measure, error_log, t)
            log_models(clients, t, exp_path)


elif args.method == 'FL-DP': #model aggregation (e.g., averaging)
    clients = []
    for _ in range(args.num_clients):
        client = client_class()
        clients.append(client)
    for t in range(args.num_rounds):
        for i,client in enumerate(clients):
            Xtrain, ytrain = dataset.getNextLocalBatch(i, args.train_batch_size)
            client.train(Xtrain, ytrain)
        if (t+1) % args.comm_period == 0:
            print("Comm-round ",(t+1))
            params = []
            for client in clients:
                client_params = client.getParameters()
                print("before...",client_params)
                normFact = max(1, (np.linalg.norm(client_params.toVector())/args.dp_s))
                print(normFact)
                client_params.scalarMultiply(1./normFact) #clipp the update
                #add noise
                clipped_update = client_params.toVector()
                clipped_update += np.random.normal(loc=0.0, scale=(args.dp_sigma**2)*(args.dp_s**2), size=clipped_update.shape)
                client_params.fromVector(clipped_update)
                print("after...",client_params)
                params.append(client_params)
            agg_params = aggregate(params)
            for client in clients:
                client.setParameters(agg_params)
            comm_log[t] = {"params":params.copy(),"agg":agg_params,"clients":params}
        if args.evaluation_rounds == 1 or (t + 1) % args.evaluation_rounds == 0:
            log_errors(clients, dataset, error_measure, error_log, t)
            log_models(clients, t, exp_path) 

elif args.method == "DD":
    clients = []
    for _ in range(args.num_clients):
        client = client_class()
        clients.append(client)
    soft_decisions = None
    for t in range(args.num_rounds):
        if isinstance(soft_decisions, list):
            soft_decisions = np.array(soft_decisions)
        soft_decisions = []
        for i, client in enumerate(clients):
            Xtrain, ytrain = dataset.getNextLocalBatch(i, args.train_batch_size)
            client.train(Xtrain, ytrain)
        if (t + 1) % args.comm_period == 0:
            print("Comm-round ", (t + 1))
            for client in clients:
                soft_decisions.append(client.predict_soft(dataset.Xunlabeled))
            soft_decisions = np.array(soft_decisions)
            args.consensus='average'
            consensus_soft_decisions = getConsensusSoftDecisions(soft_decisions,args.consensus)
            comm_log[t] = {"soft_decisions": soft_decisions.copy(), "consensus_soft_decisions": consensus_soft_decisions}
        if args.evaluation_rounds == 1 or (t + 1) % args.evaluation_rounds == 0:
            log_errors(clients, dataset, error_measure, error_log, t)
            log_models(clients, t, exp_path)
            
elif args.method == "PATE":
    clients = []
    for _ in range(args.num_clients):
        client = client_class()
        clients.append(client)

    # Training local models on local private data
    for t in range(args.num_rounds):
        for i, client in enumerate(clients):
            Xtrain, ytrain = dataset.getNextLocalBatch(i, args.train_batch_size)
            client.train(Xtrain, ytrain)
        print(" Teachers learning process ....")
        if args.evaluation_rounds == 1 or (t + 1) % args.evaluation_rounds == 0:
            log_errors(clients, dataset, error_measure, error_log, t)
            log_models(clients, t, exp_path)

    # Communication for generating pseudo labels
    predictions = np.array([client.predict(dataset.Xunlabeled) for client in clients])
    yconsensus = getConsensusLabels(predictions, args.consensus)
    
    print("Student learning process started")
    student_model = client_class()  
    student_train_errors, student_test_errors  = [], []
    # Train student model for 1000 epochs. This could be a hyperparameter as well but I just set it to 1000 for now
    for t in range(1000):
        student_model.train(dataset.Xunlabeled, yconsensus)

        # Log student errors (you might want to adjust the frequency)
        if args.evaluation_rounds == 1 or (t + 1) % args.evaluation_rounds == 0:
            ytrainpred = student_model.predict(dataset.Xunlabeled)
            student_train_errors.append(error_measure(yconsensus, ytrainpred))
            ypred = student_model.predict(dataset.Xtest)
            student_test_errors.append(error_measure(dataset.ytest, ypred))
            print("Student round", t + 1, ": student_train error", student_train_errors[-1], ", student_test_error", student_test_errors[-1])
            # Log errors at each round
            wandb.log({"student_test_acc": student_test_errors[-1], "student_train_acc": student_train_errors[-1], "student_round": t + 1})


elif args.method == "DP-PATE":
    clients = []
    for _ in range(args.num_clients):
        client = client_class()
        clients.append(client)

    epsilon = 2  # Set your desired privacy parameter
    b_teachers = 1.0 / epsilon  # Laplace noise scale for teachers

    # Training local models on local private data
    for t in range(args.num_rounds):
        for i, client in enumerate(clients):
            Xtrain, ytrain = dataset.getNextLocalBatch(i, args.train_batch_size)
            client.train(Xtrain, ytrain)
        print(" Teachers learning process ....")
        if args.evaluation_rounds == 1 or (t + 1) % args.evaluation_rounds == 0:
            log_errors(clients, dataset, error_measure, error_log, t)
            log_models(clients, t, exp_path)

    # Communication for generating pseudo labels
    # Label aggregation with privacy
    predictions = np.array([client.predict(dataset.Xunlabeled) for client in clients])
    print(predictions)

    # Add Laplace noise to each class count to introduce ambiguity
    num_classes=2
    noisy_counts_per_class = []
    for class_label in range(num_classes):  # assuming num_classes is 10 for Fashion-MNIST
        noisy_counts = add_laplace_noise(np.sum(predictions == class_label, axis=0), b_teachers)
        noisy_counts_per_class.append(noisy_counts)
    # Add Laplace noise to vote counts to introduce ambiguity
    # Stack the noisy counts for all classes
    noisy_counts = np.vstack(noisy_counts_per_class)
    # Choose the label with the highest noisy count as the consensus label
    yconsensus = np.argmax(noisy_counts, axis=0)
    print("yconsensus len ", yconsensus.shape)
    print("prediction",predictions.shape)
    print("noisy_counts", noisy_counts.shape)

    print("Student learning process started")
    student_model = client_class()  
    student_train_errors, student_test_errors  = [], []
    # Train student model for 500 epochs. This could be a hyperparameter as well but I just set it to 1000 for now
    for t in range(1000):
        student_model.train(dataset.Xunlabeled, yconsensus)

        # Log student errors (you might want to adjust the frequency)
        if args.evaluation_rounds == 1 or (t + 1) % args.evaluation_rounds == 0:
            ytrainpred = student_model.predict(dataset.Xunlabeled)
            student_train_errors.append(error_measure(yconsensus, ytrainpred))
            ypred = student_model.predict(dataset.Xtest)
            student_test_errors.append(error_measure(dataset.ytest, ypred))
            print("Student round", t + 1, ": student_train error", student_train_errors[-1], ", student_test_error", student_test_errors[-1])
            # Log errors at each round
            wandb.log({"student_test_acc": student_test_errors[-1], "student_train_acc": student_train_errors[-1], "student_round": t + 1})

else:
    print("ERROR: method not recognized: ",args.method)
    
### Logging results ###
writeLogfiles(exp_path, comm_log, error_log)

print("Experiment: done.")
