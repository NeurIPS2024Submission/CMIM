import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler

from utils import Params, NCorrect, set_logger
import logging
from models import get_network
from load_data import fetch_dataloader
from tqdm import tqdm

def trainOneEpoch(model, trainloader, lossFn, optimizer):
    model.train()
    Nsamples = 0
    Loss = 0
    NTop1=0
    with tqdm(total=len(trainloader)) as t:
        for Nbatch, (train_batch, labels_batch) in enumerate(trainloader):
            train_batch = train_batch.cuda()
            labels_batch = labels_batch.cuda()
            BatchLogits = model(train_batch)
            loss = lossFn(BatchLogits, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Nsamples += float(labels_batch.size(0))
            NTop1 += float(NCorrect(BatchLogits, labels_batch)[0])
            Loss += float(loss)
            t.update()
    return {"Acc": NTop1/Nsamples,
            "Loss": Loss/Nsamples} 
def valOneEpoch(model, valloader):
    model.eval()
    Nsamples = 0
    NTop1=0
    for data_batch, labels_batch in valloader:
        data_batch = data_batch.cuda()          # (B,3,32,32)
        labels_batch = labels_batch.cuda() 
        with torch.no_grad():
            BatchLogits = model(data_batch)
            NTop1 += float(NCorrect(BatchLogits, labels_batch)[0])
            Nsamples += float(labels_batch.size(0))
    return {"Acc": NTop1/Nsamples}

def main(params):
    if params.dataset == "cifar100":
        Nclass = 100
    elif params.dataset == "tinyimagenet":
        Nclass = 200
    model = get_network(params.model_name, Nclass)
    model = model.cuda()
    trainloader, valloader = fetch_dataloader(params)
    optimizer = SGD(model.parameters(), lr=params.learning_rate,
                    momentum=0.9, weight_decay=5e-4)
    train_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=params.schedule, gamma=params.gamma)
    lossFn = nn.CrossEntropyLoss(label_smoothing=params.LS)
    BestValAcc = -1
    BestValAccEpoch = -1
    for epoch in range(params.num_epochs):
        TrainRes = trainOneEpoch(model, trainloader, lossFn, optimizer)
        train_scheduler.step()
        #save_name = os.path.join(args.SimulationFolder, 'last_model.tar')
        #torch.save({
        #    'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optimizer.state_dict()},
        #    save_name)
        ValRes = valOneEpoch(model, valloader)
        if BestValAcc < ValRes['Acc']:
            save_name = os.path.join(args.SimulationFolder, 'best_model.tar')
            BestValAcc = ValRes['Acc']
            BestValAccEpoch = epoch
            torch.save({
                'epoch': epoch + 1, 'state_dict': model.state_dict()},
                save_name)
        logging.info('-- Epoch: {} -- Train acc: {} -- Validation Acc: {}'.format(str(epoch), TrainRes['Acc'],ValRes['Acc']))
        logging.info('Highest validation Acc: {} at epoch {}'.format(BestValAcc, str(BestValAccEpoch)))        
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--SimulationFolder', default='', required=True, type=str)
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction)
    parser.add_argument('--trial', default=1, type=int)
    args = parser.parse_args()
    params = Params(os.path.join(args.SimulationFolder, "params.json"))
    set_logger(os.path.join(args.SimulationFolder, 'training_{}.log'.format(str(args.trial))))
    set_logger(args.SimulationFolder)
    main(params)
