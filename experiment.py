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

from clients.rulefitClients import *


available_clients = getAvailableClasses(Client)
available_datasets = getAvailableClasses(Dataset)

from clients import *
from datasets import *
import aggregators
import inspect
from consensus import *
from xor import *
import copy
import random
import tensorflow as tf

torch.cuda.empty_cache()
#set the random seeds
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)

#Weight and Bises login
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
wandb.login(key='999ae259390532c6c283ea338b365a2e9b489d78')



available_aggregators = []
for name, obj in inspect.getmembers(aggregators):
    if inspect.isclass(obj):
        available_aggregators.append(name)
        
#set the parameters
parser = argparse.ArgumentParser(description='AIM-HI Experiment Suite')
parser.add_argument('--method', type=str, default='centralized',
                    choices=['centralized', 'AIMHI', 'FL', 'DP-FL','FL-DP','AIMHI-DP','DD'],
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
parser.add_argument('--distill-period', type=int, default=1,
                    help='distill period of (default: 2)')
args = parser.parse_args()    

cuda = torch.device('cuda:2')
### Initialize ###
client_class = available_clients[args.client]
num_clients = args.num_clients if args.method != "centralized" else 1
dataset = available_datasets[args.dataset](args.method, num_clients, args.num_unlabeled, args.num_samples_per_client)


if args.dataset == "CIFAR10Keras":
    name="CIFAR10_AIMHI"
if args.dataset=="MRI":
    name="MRI_AIMHI"
if args.dataset=="Pneum":
    name="Pneum_AIMHI"
if args.dataset=="FashionMNIST":
    name="FashionMNIST_AIMHI"
if args.dataset=="SVHN":
    name="SVHN_AIMHI"
if args.num_clients>5:
    name="Scalability_AIMHI"
if args.dataset=="Noniid_FashionMNIST":
    name="Noniid_FashionMNIST_AIMHI"
if args.dataset=="breast_cancer":
    name="breast_cancer_AIMHI"
if args.dataset=="winequalityN":
    name="winequalityN_AIMHI"
if args.dataset=="Heart_disease_statlog":
    name="Heart_disease_statlog_AIMHI"
#if args.method=="DP-FL" or "FL-DP":
 #   name="FL-DP"
if args.method=="AIMHI-DP":
    name="AIMHI-DP"
if args.method=="DD":
    name="DD"
args.consensus='average'
# Start a run, tracking hyperparameters
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
        "consensus": args.consensus
        
        
    }
)

config = wandb.config





aggregate_class = getattr(aggregators, args.aggregator)
aggregate = aggregate_class()

error_measure = getErrorMeasure(args.error)
exp_path = getExpPath(args)

dataset.saveDatasetIndices(exp_path)

AIMHI_EXPORT_STATEDICT = True

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
elif args.method == "AIMHI": #label sharing
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

elif args.method == "AIMHI-DP": #label sharing
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
                prediction_binary=torch.eye(10)[client.predict(dataset.Xunlabeled)].type(torch.int64)
                #print("prediction binray issss",prediction_binary)
                #prediction_binary=xor_mechanism(prediction_binary,args.dp_s,prediction_binary)
                print("dps is .......",args.dp_s)
                prediction_binary=xor_noise(prediction_binary,args.dp_s)
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
elif args.method == "FL": #model aggregation (e.g., averaging)
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
                params.append(client.getParameters())
            agg_params = aggregate(params)
            for client in clients:
                client.setParameters(agg_params)
            comm_log[t] = {"params":params.copy(),"agg":agg_params,"clients":params}
        if args.evaluation_rounds == 1 or (t + 1) % args.evaluation_rounds == 0:
            log_errors(clients, dataset, error_measure, error_log, t)
            log_models(clients, t, exp_path)

elif args.method == 'DP-FL':
    clients = []
    for _ in range(args.num_clients):
        client = client_class()
        clients.append(client)
    for t in range(args.num_rounds):
        for i, client in enumerate(clients):
            Xtrain, ytrain = dataset.getNextLocalBatch(i, args.train_batch_size)
            client.train(Xtrain, ytrain)
        if (t + 1) % args.comm_period == 0:
            print("Comm-round ", (t + 1))
            params = []
            for client in clients:
                client_params = client.getParameters().getCopy()
                print("before: ", np.linalg.norm(client_params.toVector()))
                normFact = max(1, (np.linalg.norm(client_params.toVector()) / args.dp_s))
                print("norm fact: ", normFact)
                client_params.scalarMultiply(1. / normFact)  # clip the update
                update = client_params.toVector() - client.getParameters().toVector()
                # add noise to update
                noise = np.random.normal(loc=0.0, scale=(args.dp_sigma ** 2) * (args.dp_s ** 2), size=update.shape)
                clipped_update = np.clip(update, -args.dp_s, args.dp_s)
                noisy_update = clipped_update + noise
                client_params.fromVector(client.getParameters().toVector() + noisy_update)
                print("noise: ", np.linalg.norm(noise))
                print("after: ", np.linalg.norm(client_params.toVector()))
                params.append(client_params)
            agg_params = aggregate(params)
            for client in clients:
                client.setParameters(agg_params)
            comm_log[t] = {"params": params.copy(), "agg": agg_params, "clients": params}
        if args.evaluation_rounds == 1 or (t + 1) % args.evaluation_rounds == 0:
            log_models(clients, t, exp_path)
            log_errors(clients, dataset, error_measure, error_log, t)

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

elif args.method == "DD-ODL": # Distributed Distillation for On-Device Learning
    teacher = teacher_class()
    clients = []
    for _ in range(args.num_clients):
        client = client_class()
        clients.append(client)
    for t in range(args.rounds):
        if (t+1) % args.distill_period == 0:
            teacher.train(dataset.Xtrain, dataset.ytrain)
        for i, client in enumerate(clients):
            Xtrain, ytrain = dataset.getNextLocalBatch(i, args.batch_size)
            if (t+1) % args.distill_period == 0:
                ytrain = teacher.predict(Xtrain)
            client.train(Xtrain, ytrain)
        if (t+1) % args.communication_period == 0:
            models = []
            for client in clients:
                models.append(client.getParameters())
            agg_params = aggregate(models, args.aggregation_method)
            for client in clients:
                client.setParameters(agg_params)
        if args.evaluation_rounds == 1 or (t+1) % args.evaluation_rounds == 0:
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


else:
    print("ERROR: method not recognized: ",args.method)
    
### Logging results ###
writeLogfiles(exp_path, comm_log, error_log)

print("Experiment: done.")


        
            