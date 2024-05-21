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

class APGP_Layer(nn.Module):
    def __init__(self, num_class=100):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_class, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_class)
        )
    def forward(self, x):
        output = self.classifier(x)+x
        return output

class APGPLoss():
    def __init__(self, nclass=100, Batchsize = 128, lmbda = 0.1, reduction = 'batchmean') -> None:
        self.lfn1 = nn.KLDivLoss(reduction=reduction, log_target=True)
        self.Cidx = ((torch.arange(Batchsize)[None,:].expand(nclass,-1).T)*nclass).cuda()
        self.lmbda = lmbda
        self.CEloss = nn.CrossEntropyLoss()
    def __call__(self,preds, APGP_preds, target):
        # breakpoint()
        loss1 = - self.lfn1(F.log_softmax(APGP_preds, dim=1), F.log_softmax(preds, dim=1))
        loss2 =  self.CEloss(APGP_preds, target)

        loss = loss2 + self.lmbda* loss1
        return loss

def trainOneEpoch(Teacher, APGPLayer, lossfn, trainloader, optimizer):
    # (model, Teacher, trainloader, CElossFn, KLlossFn, optimizer, alpha):
    APGPLayer.train()
    Teacher.eval()
    Nsamples = 0
    Loss = 0
    NTop1=0
    with tqdm(total=len(trainloader)) as t:
        for Nbatch, (train_batch, labels_batch) in enumerate(trainloader):
            train_batch = train_batch.cuda()
            labels_batch = labels_batch.cuda()
            
            with torch.no_grad():
                TeacherLogits = Teacher(train_batch)
            BatchLogits = APGPLayer(F.softmax(TeacherLogits,1))
            
            loss = lossfn(TeacherLogits, BatchLogits, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Nsamples += float(labels_batch.size(0))
            NTop1 += float(NCorrect(BatchLogits, labels_batch)[0])
            Loss += float(loss)
            t.update()
    return {"Acc": NTop1/Nsamples,
            "Loss": Loss/Nsamples} 
def valOneEpoch(model,APGPLayer, valloader):
    model.eval()
    Nsamples = 0
    NTop1=0
    for data_batch, labels_batch in tqdm(valloader):
        data_batch = data_batch.cuda()          # (B,3,32,32)
        labels_batch = labels_batch.cuda() 
        with torch.no_grad():
            BatchLogits = APGPLayer(F.softmax(model(data_batch), 1))
            NTop1 += float(NCorrect(BatchLogits, labels_batch)[0])
            Nsamples += float(labels_batch.size(0))
    return {"Acc": NTop1/Nsamples}

def main(params):
    if params.dataset == "cifar100":
        Nclass = 100
    elif params.dataset == "tinyimagenet":
        Nclass = 200
    Teacher = get_network(params.teacherModel, Nclass)
    checkpoint = torch.load(params.TeacherCkpt)
    Teacher.load_state_dict(checkpoint['state_dict'])
    Teacher = Teacher.cuda()
    Teacher.eval()
    APGPLayer = APGP_Layer(Nclass).cuda()
    trainloader, valloader = fetch_dataloader(params)
    optimizer = SGD(APGPLayer.parameters(), lr=params.learning_rate,
                    momentum=0.9, weight_decay=5e-4)
    train_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=params.schedule, gamma=params.gamma)
    lossfn = APGPLoss( nclass=Nclass, Batchsize = params.batch_size, lmbda = params.lmbda)

    for epoch in range(params.num_epochs):
        TrainRes = trainOneEpoch(Teacher, APGPLayer, lossfn, trainloader, optimizer)
        print(TrainRes)
        train_scheduler.step()
        ValRes = valOneEpoch(Teacher, APGPLayer, valloader)
        print(ValRes)

    save_name = os.path.join(args.SimulationFolder, 'APGPLayer.tar')

    torch.save({ 'state_dict': APGPLayer.state_dict()}, save_name)
# logging.info('-- Epoch: {} -- Train acc: {} -- Validation Acc: {}'.format(str(epoch), TrainRes['Acc'],ValRes['Acc']))
# logging.info('Highest validation Acc: {} at epoch {}'.format(BestValAcc, str(BestValAccEpoch)))        
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
