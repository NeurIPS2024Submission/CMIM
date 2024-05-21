import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from torch.optim import Adam
from utils import Params, NCorrect, set_logger
import logging
from models import get_network
from load_data import fetch_dataloader
from tqdm import tqdm


class Stingy(nn.Module):
    def __init__(self, TopN=10):
        super().__init__()
        self.TopN = TopN
    def forward(self, Prob):
        mask = torch.zeros_like(Prob)
        indices = torch.topk(Prob, self.TopN).indices.cuda()
        mask = mask.scatter_(1, indices, 1.)
        Prob = mask*Prob
        rt = Prob/Prob.sum(1)[:,None]
        return rt


class CATModel(nn.Module):
    def __init__(self, Nclass=10, intrinct_dim=3, params=None):
        super().__init__()
        Weights = []
        self.tempTrans = 1
        if params:
            self.tempTrans = params.temperature
        for _ in range(Nclass):
            Weight = torch.nn.init.kaiming_uniform_(torch.ones(Nclass, 2, intrinct_dim))
            Weight = torch.transpose(Weight, 0,1)
            Weights.append(nn.Parameter(Weight))
        self.params = nn.ParameterList(Weights)
        self.Iscale = torch.nn.Parameter(torch.ones(Nclass)) #*math.log(alpha*(Nclass-1)/(1-alpha))
        self.I = torch.eye(Nclass).cuda()
    def forward(self, TeacherLogit, Target):
        TeacherProb = TeacherLogit**(1/self.tempTrans)
        TeacherProb = TeacherProb/TeacherProb.sum(1)[:,None]
        rtProb = torch.zeros_like(TeacherProb)
        UniqueTargets = torch.unique(Target)
        for UniqueTarget in UniqueTargets:
            mask = Target==UniqueTarget
            rtProb[mask] = torch.mm(TeacherProb[mask], F.softmax(self.I*self.Iscale[UniqueTarget]+(self.params[UniqueTarget][0]@torch.transpose(self.params[UniqueTarget][1], 0,1)), dim=0))
        return rtProb


class KLLossfnWithT():
    def __init__(self, T=4, reduction = 'mean') -> None:
        self.T = T
        self.lfn = nn.KLDivLoss(reduction=reduction)
    def __call__(self,preds, teacher_preds):
        loss = self.lfn(F.log_softmax(preds/self.T, dim=1),
                        F.softmax(teacher_preds/self.T, dim=1)) * (self.T * self.T)
        return loss

def trainOneEpoch(model, Teacher,Stingylayer, CatLayer, trainloader, CElossFn, KLlossFn, optimizer, CAToptimizer, params, alpha):
    model.train()
    Teacher.eval()
    Nsamples = 0
    Loss = 0
    NTop1=0
    T = params.temperature
    with tqdm(total=len(trainloader)) as t:
        for Nbatch, (train_batch, labels_batch) in enumerate(trainloader):
            train_batch = train_batch.cuda()
            labels_batch = labels_batch.cuda()
            BatchLogits = model(train_batch)
            with torch.no_grad():
                TeacherLogits = Stingylayer(F.softmax(Teacher(train_batch), 1))
            Transfered_output_teacher_batch = CatLayer(TeacherLogits, labels_batch)
            hardLoss = CElossFn(BatchLogits, labels_batch)
            softLoss = KLlossFn(BatchLogits, Transfered_output_teacher_batch.detach())
            loss = (1-alpha)*hardLoss +  alpha*softLoss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            CATloss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax((BatchLogits.detach())/T, dim=1),
                             F.softmax(Transfered_output_teacher_batch/T, dim=1)) * (alpha * T * T)
            CAToptimizer.zero_grad()
            CATloss.backward()
            CAToptimizer.step()
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
        TopN = 10
    elif params.dataset == "tinyimagenet":
        Nclass = 200
        TopN = 20
    model = get_network(params.model_name, Nclass)
    Teacher = get_network(params.teacherModel, Nclass)
    checkpoint = torch.load(params.TeacherCkpt)
    Teacher.load_state_dict(checkpoint['state_dict'])
    Teacher = Teacher.cuda()
    Teacher.eval()
    model = model.cuda()
    trainloader, valloader = fetch_dataloader(params)
    optimizer = SGD(model.parameters(), lr=params.learning_rate,
                    momentum=0.9, weight_decay=5e-4)

    CatLayer = CATModel(Nclass=Nclass, intrinct_dim=3, params=params).cuda()

    CAToptimizer = Adam(CatLayer.parameters(), lr=params.learning_rate/100.)
    train_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=params.schedule, gamma=params.gamma)
    CElossFn = nn.CrossEntropyLoss()
    KLlossFn = KLLossfnWithT(T=params.temperature, reduction='mean')
    alpha = params.alpha
    BestValAcc = -1
    BestValAccEpoch = -1
    Stingylayer = Stingy(TopN).cuda()
    # ValRes = valOneEpoch(Teacher, valloader)
    # logging.info('Teacher validation Acc: {}'.format(ValRes['Acc']))
    # ValRes = valOneEpoch(Teacher, trainloader)
    # logging.info('Teacher training  Acc: {}'.format(ValRes['Acc']))

    for epoch in range(params.num_epochs):
        # trainOneEpoch(model, Teacher, CatLayer, trainloader, CElossFn, KLlossFn, optimizer, CAToptimizer, params, alpha)
        TrainRes = trainOneEpoch(model, Teacher,Stingylayer, CatLayer, trainloader, CElossFn, KLlossFn, optimizer, CAToptimizer, params, alpha)
        train_scheduler.step()
        # save_name = os.path.join(args.SimulationFolder, 'last_model.tar')
        # torch.save({
        #    'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optimizer.state_dict()},
        #    save_name)
        ValRes = valOneEpoch(model, valloader)
        if BestValAcc < ValRes['Acc']:
            save_name = os.path.join(args.SimulationFolder, 'best_model.tar')
            torch.save({
           'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optimizer.state_dict()},
           save_name)
            BestValAcc = ValRes['Acc']
            BestValAccEpoch = epoch

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
