# train a student network distilling from teacher by DKD
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

class KLLossfnWithT():
    def __init__(self, T=4, reduction = 'mean') -> None:
        self.T = T
        self.lfn = nn.KLDivLoss(reduction=reduction)
    def __call__(self,preds, teacher_preds):
        loss = self.lfn(F.log_softmax(preds/self.T, dim=1),
                        F.softmax(teacher_preds/self.T, dim=1)) * (self.T * self.T)
        return loss

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature, TopN=10):
    mask = torch.zeros_like(logits_teacher)
    indices = torch.topk(logits_teacher, TopN).indices.cuda()
    mask = mask.scatter_(1, indices, 1)
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)*mask
    pred_teacher = pred_teacher/pred_teacher.sum(1)[:,None]
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )*mask
    pred_teacher_part2 = pred_teacher_part2/pred_teacher_part2.sum(1)[:,None]

    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def trainOneEpoch(model, Teacher, trainloader, optimizer, alpha, beta, temperature, TopN):
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
                TeacherLogits = Teacher(train_batch)

            loss = dkd_loss(BatchLogits, TeacherLogits, labels_batch, alpha, beta, temperature, TopN)
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

    alpha = params.alpha
    beta = params.beta
    temperature = params.Temp

    BestValAcc = -1
    BestValAccEpoch = -1

    ValRes = valOneEpoch(Teacher, valloader)
    logging.info('Teacher validation Acc: {}'.format(ValRes['Acc']))
    ValRes = valOneEpoch(Teacher, trainloader)
    logging.info('Teacher training  Acc: {}'.format(ValRes['Acc']))

    for epoch in range(params.num_epochs):
        TrainRes = trainOneEpoch(model, Teacher, trainloader, optimizer, alpha, beta, temperature, TopN)
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
