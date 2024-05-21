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

class KLLossfnWithT():
    def __init__(self, T=4, reduction = 'mean') -> None:
        self.T = T
        self.lfn = nn.KLDivLoss(reduction=reduction)
    def __call__(self,preds, teacher_Prob):
        teacher_Prob = teacher_Prob**(1/self.T)
        teacher_Prob = teacher_Prob/torch.sum(teacher_Prob,1)[:,None]
        loss = self.lfn(F.log_softmax(preds/self.T, dim=1),
                        teacher_Prob) * (self.T * self.T)
        return loss


def trainOneEpoch(model, Teacher, Stingylayer, trainloader, CElossFn, KLlossFn, optimizer, alpha):
    model.train()
    Teacher.eval()
    Nsamples = 0
    Loss = 0
    NTop1=0
    with tqdm(total=len(trainloader)) as t:
        for Nbatch, (train_batch, labels_batch) in enumerate(trainloader):
            train_batch = train_batch.cuda()
            labels_batch = labels_batch.cuda()
            BatchLogits = model(train_batch)
            with torch.no_grad():
                # TeacherLogits = Teacher(train_batch)
                TeacherLogits = Stingylayer(F.softmax(Teacher(train_batch), 1))

            hardLoss = CElossFn(BatchLogits, labels_batch)
            softLoss = KLlossFn(BatchLogits, TeacherLogits)
            loss = (1-alpha)*hardLoss +  alpha*softLoss

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
    train_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=params.schedule, gamma=params.gamma)
    CElossFn = nn.CrossEntropyLoss()
    KLlossFn = KLLossfnWithT(T=params.Temp, reduction='mean')
    alpha = params.alpha
    BestValAcc = -1
    BestValAccEpoch = -1
    Stingylayer = Stingy(TopN).cuda()

    for epoch in range(params.num_epochs):
        TrainRes = trainOneEpoch(model, Teacher, Stingylayer, trainloader, CElossFn, KLlossFn, optimizer, alpha)
        train_scheduler.step()
        #save_name = os.path.join(args.SimulationFolder, 'last_model.tar')
        #torch.save({
        #    'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optim_dict': optimizer.state_dict()},
        #    save_name)
        ValRes = valOneEpoch(model, valloader)
        if BestValAcc < ValRes['Acc']:
            # save_name = os.path.join(args.SimulationFolder, 'best_model.tar')
            BestValAcc = ValRes['Acc']
            BestValAccEpoch = epoch
            # torch.save({
            #     'epoch': epoch + 1, 'state_dict': model.state_dict()},
            #     save_name)
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
