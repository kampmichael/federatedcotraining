#! /bin/bash --


# Select a method
method="AIMHI"
# set number of clients m
numclients=5
# number of rounds T
numrounds=20000
# communication period b
commperiod=50
# Evaluation period
evaluationrounds=50
# number of unlabled data point U
numunlabeled=50000
# train batch size
trainbatchsize=64
# select a client
clients='PaperPytorchFashionMNIST'
# select a dataset
dataset="FashionMNIST"

# select a cuda device
cuda_device=0

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${cuda_device} python3 -u experiment.py --method $method --client $clients --dataset $dataset --num-clients $numclients --num-rounds $numrounds --train-batch-size $trainbatchsize --comm-period $commperiod --evaluation-rounds $evaluationrounds --num-unlabeled $numunlabeled