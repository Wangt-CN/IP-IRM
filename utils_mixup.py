from PIL import Image
from torchvision import transforms
from torchvision.datasets import STL10, CIFAR10, CIFAR100
import cv2
import numpy as np
import torch
import math
import torch.nn.functional as F
from torch import nn, optim, autograd
from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR
from torch.utils.data import DataLoader
from torch.utils import data
import random
from tqdm import tqdm

np.random.seed(0)


def info_nce_loss_formixup(q, k, temperature):
    logits = q.mm(k.t()) / temperature
    return logits



def penalty(logits, y, loss_function, mode='w', batchsize=None):
    if mode == 'w':
        scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
        try:
            loss = loss_function(logits * scale, y)
        except:
            assert batchsize is not None
            loss = loss_function(logits * scale, y, batchsize)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
    elif mode == 'f':
        pass
    return torch.sum(grad**2)


class update_split_dataset(data.Dataset):
    def __init__(self, feature_bank1, feature_bank2):
        """Initialize and preprocess the Dsprite dataset."""
        self.feature_bank1 = feature_bank1
        self.feature_bank2 = feature_bank2


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        feature1 = self.feature_bank1[index]
        feature2 = self.feature_bank2[index]

        return feature1, feature2, index

    def __len__(self):
        """Return the number of images."""
        return self.feature_bank1.size(0)


# Update split online with mixup
def auto_split_online_mixup(net, update_loader, soft_split_all, temperature, irm_temp, args, loss_mode='v2', irm_mode='v1', irm_weight=10, constrain=False, cons_relax=False, nonorm=False, log_file=None):
    # irm mode: v1 is original irm; v2 is variance

    low_loss, constrain_loss = 1e5, torch.Tensor([0.])
    cnt, best_epoch, training_num = 0, 0, 0
    num_env = soft_split_all.size(1)

    # optimizer and schedule
    pre_optimizer = torch.optim.Adam([soft_split_all], lr=0.1, weight_decay=0.)
    pre_scheduler = MultiStepLR(pre_optimizer, [5, 25], gamma=0.2, last_epoch=-1)

    # dataset and dataloader
    for epoch in range(40):
        risk_all_list, risk_cont_all_list, risk_penalty_all_list, risk_constrain_all_list, training_num = [],[],[],[], 0
        net.eval()
        for batch_idx, (pos_1, pos_2, target, idx) in enumerate(update_loader):
            training_num += len(pos_1)
            with torch.no_grad():
                pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
                bsz = pos_1.shape[0]
                pos_1_mixup, labels_aux, lam = mixup(pos_1, args.alpha)

                _, feature_1 = net(pos_1_mixup)
                _, feature_2 = net(pos_2)

            loss_cont_list, loss_penalty_list = [], []

            # Option 1. use soft split
            param_split = F.softmax(soft_split_all[idx], dim=-1)
            if irm_mode == 'v1': # original
                for env_idx in range(num_env):
                    logits = feature_1.mm(feature_2.t()) / args.temperature
                    labels = torch.arange(bsz, dtype=torch.long).cuda()

                    loss_weight = param_split[:, env_idx]
                    cont_loss_env = soft_contrastive_loss_mixup_online(logits, labels, loss_weight, labels_aux, lam, mode=loss_mode, nonorm=nonorm)
                    scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
                    cont_loss_env_scale = soft_contrastive_loss_mixup_online(logits*scale, labels, loss_weight, labels_aux, lam, mode=loss_mode, nonorm=nonorm)
                    penalty_irm = torch.autograd.grad(cont_loss_env_scale, [scale], create_graph=True)[0]
                    loss_cont_list.append(cont_loss_env)
                    loss_penalty_list.append(torch.sum(penalty_irm**2))

                cont_loss_epoch = torch.stack(loss_cont_list).mean()
                inv_loss_epoch = torch.stack(loss_penalty_list).mean()
                risk_final = - (cont_loss_epoch + irm_weight*inv_loss_epoch)

            else:
                raise NotImplementedError

            if constrain:
                if nonorm:
                    constrain_loss = 0.2*(- cal_entropy(param_split.mean(0), dim=0) + cal_entropy(param_split, dim=1).mean())
                else:
                    if cons_relax: # relax constrain to make item num of groups no more than 2:1
                        constrain_loss = torch.relu(0.6365 - cal_entropy(param_split.mean(0), dim=0))
                    else:
                        constrain_loss = - cal_entropy(param_split.mean(0), dim=0)#  + cal_entropy(param_split, dim=1).mean()
                risk_final += constrain_loss


            pre_optimizer.zero_grad()
            risk_final.backward()
            pre_optimizer.step()

            risk_all_list.append(risk_final.item())
            risk_cont_all_list.append(-cont_loss_epoch.item())
            risk_penalty_all_list.append(-inv_loss_epoch.item())
            risk_constrain_all_list.append(constrain_loss.item())
            soft_split_print = soft_split_all[:1].clone().detach()
            if epoch > 0:
                print('\rUpdating Env [%d/%d] [%d/%d]  Loss: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2e  Cons_Risk: %.2f  Cnt: %d  Lr: %.4f  Inv_Mode: %s  Soft Split: %s'
                      %(epoch, 30, training_num, len(update_loader.dataset), sum(risk_all_list)/len(risk_all_list), sum(risk_cont_all_list)/len(risk_cont_all_list), sum(risk_penalty_all_list)/len(risk_penalty_all_list),
                        sum(risk_constrain_all_list)/len(risk_constrain_all_list), cnt, pre_optimizer.param_groups[0]['lr'], irm_mode, F.softmax(soft_split_print, dim=-1)), end='', flush=True)


        pre_scheduler.step()
        avg_risk = sum(risk_all_list)/len(risk_all_list)
        avg_cont_risk = sum(risk_cont_all_list)/len(risk_cont_all_list)
        avg_inv_risk = sum(risk_penalty_all_list)/len(risk_penalty_all_list)

        if epoch == 0:
            write_log("Initial Risk: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2e" %(avg_risk, avg_cont_risk, avg_inv_risk), log_file=log_file, print_=True)
            soft_split_best = soft_split_all.clone().detach()
        if avg_risk < low_loss:
            low_loss = avg_risk
            soft_split_best = soft_split_all.clone().detach()
            best_epoch = epoch
            cnt = 0
        else:
            cnt += 1


        if epoch > 25 and cnt >= 5 or epoch == 30: #debug
            write_log('\nLoss not down. Break down training.  Epoch: %d  Loss: %.2f' %(best_epoch, low_loss), log_file=log_file, print_=True)
            write_log('Updating Env [%d/%d] [%d/%d]  Loss: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2e  Cons_Risk: %.2f  Cnt: %d  Lr: %.4f  Inv_Mode: %s'
                      %(epoch, 100, training_num, len(update_loader.dataset), sum(risk_all_list)/len(risk_all_list), sum(risk_cont_all_list)/len(risk_cont_all_list), sum(risk_penalty_all_list)/len(risk_penalty_all_list),
                        sum(risk_constrain_all_list)/len(risk_constrain_all_list), cnt, pre_optimizer.param_groups[0]['lr'], irm_mode), log_file=log_file)
            final_split_softmax = F.softmax(soft_split_best, dim=-1)
            write_log('%s' %(final_split_softmax), log_file=log_file, print_=True)
            group_assign = final_split_softmax.argmax(dim=1)
            write_log('Debug:  group1 %d  group2 %d' %(group_assign.sum(), group_assign.size(0)-group_assign.sum()), log_file=log_file, print_=True)
            return soft_split_best



def auto_split_offline_mixuup(out_1, out_2, labels_aux_all, lam_all, soft_split_all, temperature, irm_temp, loss_mode='v2', irm_mode='v1', irm_weight=10, constrain=False, cons_relax=False, nonorm=False, log_file=None):
    # irm mode: v1 is original irm; v2 is variance
    low_loss, constrain_loss = 1e5, torch.Tensor([0.])
    cnt, best_epoch, training_num = 0, 0, 0
    num_env = soft_split_all.size(1)
    # optimizer and schedule
    pre_optimizer = torch.optim.Adam([soft_split_all], lr=0.5, weight_decay=0.)
    pre_scheduler = MultiStepLR(pre_optimizer, [5, 35], gamma=0.2, last_epoch=-1)
    # dataset and dataloader
    traindataset = update_split_dataset(out_1, out_2)
    trainloader = DataLoader(traindataset, batch_size=2048, shuffle=True, num_workers=4)
    for epoch in range(100):
        risk_all_list, risk_cont_all_list, risk_penalty_all_list, risk_constrain_all_list, training_num = [],[],[],[], 0

        for feature_1, feature_2, idx in trainloader:
            feature_1, feature_2 = feature_1.cuda(), feature_2.cuda()
            loss_cont_list, loss_penalty_list = [], []
            training_num += len(feature_1)
            # Option 1. use soft split
            param_split = F.softmax(soft_split_all[idx], dim=-1)
            if irm_mode == 'v1': # original
                for env_idx in range(num_env):
                    logits_all = feature_1.mm(feature_2.t()) / temperature
                    bsz = feature_1.shape[0]
                    labels_all = torch.arange(bsz, dtype=torch.long).cuda()

                    loss_weight = param_split[:, env_idx]
                    cont_loss_env = soft_contrastive_loss_mixup_offline(logits_all, labels_all, loss_weight, labels_aux_all[idx], lam_all[idx], mode=loss_mode, nonorm=nonorm)

                    scale = torch.ones((1, logits_all.size(-1))).cuda().requires_grad_()
                    cont_loss_env_scale = soft_contrastive_loss_mixup_offline(logits_all*scale, labels_all, loss_weight, labels_aux_all[idx], lam_all[idx], mode=loss_mode, nonorm=nonorm)
                    penalty_irm = torch.autograd.grad(cont_loss_env_scale, [scale], create_graph=True)[0]
                    loss_cont_list.append(cont_loss_env)
                    loss_penalty_list.append(torch.sum(penalty_irm**2))

                cont_loss_epoch = torch.stack(loss_cont_list).mean()
                inv_loss_epoch = torch.stack(loss_penalty_list).mean()
                risk_final = - (cont_loss_epoch + irm_weight*inv_loss_epoch)


            else:
                raise NotImplementedError

            if constrain:
                if nonorm:
                    constrain_loss = 0.2*(- cal_entropy(param_split.mean(0), dim=0) + cal_entropy(param_split, dim=1).mean())
                else:
                    if cons_relax: # relax constrain to make item num of groups no more than 2:1
                        constrain_loss = torch.relu(0.6365 - cal_entropy(param_split.mean(0), dim=0))
                    else:
                        constrain_loss = - cal_entropy(param_split.mean(0), dim=0)#  + cal_entropy(param_split, dim=1).mean()
                risk_final += constrain_loss

            pre_optimizer.zero_grad()
            risk_final.backward()
            pre_optimizer.step()

            risk_all_list.append(risk_final.item())
            risk_cont_all_list.append(-cont_loss_epoch.item())
            risk_penalty_all_list.append(-inv_loss_epoch.item())
            risk_constrain_all_list.append(constrain_loss.item())
            soft_split_print = soft_split_all[:1].clone().detach()
            if epoch > 0:
                print('\rUpdating Env [%d/%d] [%d/%d]  Loss: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2e  Cons_Risk: %.2f  Cnt: %d  Lr: %.4f  Inv_Mode: %s  Soft Split: %s'
                      %(epoch, 100, training_num, len(trainloader.dataset), sum(risk_all_list)/len(risk_all_list), sum(risk_cont_all_list)/len(risk_cont_all_list), sum(risk_penalty_all_list)/len(risk_penalty_all_list),
                        sum(risk_constrain_all_list)/len(risk_constrain_all_list), cnt, pre_optimizer.param_groups[0]['lr'], irm_mode, F.softmax(soft_split_print, dim=-1)), end='', flush=True)

        pre_scheduler.step()
        avg_risk = sum(risk_all_list)/len(risk_all_list)
        avg_cont_risk = sum(risk_cont_all_list)/len(risk_cont_all_list)
        avg_inv_risk = sum(risk_penalty_all_list)/len(risk_penalty_all_list)

        if epoch == 0:
            write_log("Initial Risk: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2e" % (avg_risk, avg_cont_risk, avg_inv_risk), log_file=log_file, print_=True)
            soft_split_best = soft_split_all.clone().detach()
        if avg_risk < low_loss:
            low_loss = avg_risk
            soft_split_best = soft_split_all.clone().detach()
            best_epoch = epoch
            cnt = 0
        else:
            cnt += 1

        if epoch > 50 and cnt >= 5 or epoch == 60: #debug
        # if epoch > 20:
            write_log('\nLoss not down. Break down training.  Epoch: %d  Loss: %.2f' %(best_epoch, low_loss), log_file=log_file, print_=True)
            write_log('Updating Env [%d/%d] [%d/%d]  Loss: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2e  Cons_Risk: %.2f  Cnt: %d  Lr: %.4f  Inv_Mode: %s'
                      %(epoch, 100, training_num, len(trainloader.dataset), sum(risk_all_list)/len(risk_all_list), sum(risk_cont_all_list)/len(risk_cont_all_list), sum(risk_penalty_all_list)/len(risk_penalty_all_list),
                        sum(risk_constrain_all_list)/len(risk_constrain_all_list), cnt, pre_optimizer.param_groups[0]['lr'], irm_mode), log_file=log_file)
            final_split_softmax = F.softmax(soft_split_best, dim=-1)
            write_log('%s' %(final_split_softmax), log_file=log_file, print_=True)
            group_assign = final_split_softmax.argmax(dim=1)
            write_log('Debug:  group1 %d  group2 %d' %(group_assign.sum(), group_assign.size(0)-group_assign.sum()), log_file=log_file, print_=True)
            return soft_split_best



# soft version of contrastive loss for mixup offline
def soft_contrastive_loss_mixup_offline(logits, labels, weights, labels_aux, lam, mode='v1', nonorm=False):
    if mode == 'v1':
        logits *= weights
        cont_loss_env = torch.nn.CrossEntropyLoss()(logits, labels)
    elif mode == 'v2':
        sample_dim, label_dim = logits.size(0), logits.size(1)
        logits_exp = logits.exp()
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(logits.device)
        weights = weights.unsqueeze(0).repeat(sample_dim, 1)
        weight_pos = weights[mask]
        weights_mask = weights * (~mask)

        weight_neg_norm = weights_mask / weights_mask.sum(1).unsqueeze(1) * (label_dim - 1)
        weights_new = mask + weight_neg_norm
        softmax_loss = (weights_new*logits_exp) / (weights_new*logits_exp).sum(1).unsqueeze(1)
        cont_loss_env = lam * torch.nn.NLLLoss(reduction='none')(torch.log(softmax_loss), labels)
        if nonorm:
            cont_loss_env = (cont_loss_env * weight_pos.squeeze()).sum() / sample_dim
        else:
            cont_loss_env = (cont_loss_env * weight_pos.squeeze()).sum() / weight_pos.sum()    # norm version

    return cont_loss_env


# soft version of contrastive loss for mixup online
def soft_contrastive_loss_mixup_online(logits, labels, weights, labels_aux, lam, mode='v1', nonorm=False):
    if mode == 'v1':
        logits *= weights
        cont_loss_env = torch.nn.CrossEntropyLoss()(logits, labels)
    elif mode == 'v2':
        sample_dim, label_dim = logits.size(0), logits.size(1)
        logits_exp = logits.exp()
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(logits.device)
        weights = weights.unsqueeze(0).repeat(sample_dim, 1)
        weight_pos = weights[mask]
        weights_mask = weights * (~mask)

        weight_neg_norm = weights_mask / weights_mask.sum(1).unsqueeze(1) * (label_dim - 1)
        weights_new = mask + weight_neg_norm
        softmax_loss = (weights_new*logits_exp) / (weights_new*logits_exp).sum(1).unsqueeze(1)
        cont_loss_env = (lam * torch.nn.NLLLoss(reduction='none')(torch.log(softmax_loss), labels) + (1. - lam) * torch.nn.NLLLoss(reduction='none')(torch.log(softmax_loss), labels_aux)).mean()
        if nonorm:
            cont_loss_env = (cont_loss_env * weight_pos.squeeze()).sum() / sample_dim
        else:
            cont_loss_env = (cont_loss_env * weight_pos.squeeze()).sum() / weight_pos.sum()    # norm version

    return cont_loss_env



class update_split_dataset(data.Dataset):
    def __init__(self, feature_bank1, feature_bank2):
        """Initialize and preprocess the Dsprite dataset."""
        self.feature_bank1 = feature_bank1
        self.feature_bank2 = feature_bank2


    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        feature1 = self.feature_bank1[index]
        feature2 = self.feature_bank2[index]

        return feature1, feature2, index

    def __len__(self):
        """Return the number of images."""
        return self.feature_bank1.size(0)


def assign_samples(data, split, env_idx):
    # data: 2048
    images_pos1, images_pos2, labels, idxs = data
    group_assign = split[idxs].argmax(dim=1)
    select_idx = torch.where(group_assign==env_idx)[0]
    return images_pos1[select_idx], images_pos2[select_idx]

def assign_features(feature1, feature2, idxs, split, env_idx):
    group_assign = split[idxs].argmax(dim=1)
    select_idx = torch.where(group_assign==env_idx)[0]
    return feature1[select_idx], feature2[select_idx]

def assign_idxs(idxs, split, env_idx):
    group_assign = split[idxs].argmax(dim=1)
    select_idx = torch.where(group_assign==env_idx)[0]
    return select_idx


def cal_entropy(prob, dim=1):
    return -(prob * prob.log()).sum(dim=dim)


def irm_scale(irm_loss, default_scale=-100):
    with torch.no_grad():
        scale =  default_scale / irm_loss.clone().detach()
    return scale



def inputmix(input, alpha, num_aux=1, pmin=.5, distributed=False):

    bsz = input.shape[0]
    if not isinstance(alpha, (list, tuple)):
        alpha = [alpha] * (num_aux+1)
    if num_aux > 1:
        dist = torch.distributions.dirichlet.Dirichlet(torch.tensor(alpha))
        output = torch.zeros_like(input)
        lam = dist.sample([bsz]).t().to(device=input.device)
        lam = pmin * lam
        lam[0] = lam[0] + pmin
        for i in range(num_aux+1):
            if i == 0:
                randind = torch.arange(bsz, device=input.device)
            else:
                randind = torch.randperm(bsz, device=input.device)
            lam_expanded = lam[i].view([-1] + [1]*(input.dim()-1))
            output += lam_expanded * input[randind]
    else:
        beta = torch.distributions.beta.Beta(*alpha)
        randind = torch.randperm(bsz, device=input.device)
        lam = beta.sample([bsz]).to(device=input.device)
        lam = torch.max(lam, 1. - lam)
        lam_expanded = lam.view([-1] + [1]*(input.dim()-1))
        output = lam_expanded * input + (1. - lam_expanded) * input[randind]

    return output


def mixup(input, alpha, share_lam=False):
    if not isinstance(alpha, (list, tuple)):
        alpha = [alpha, alpha]
    beta = torch.distributions.beta.Beta(*alpha)
    randind = torch.randperm(input.shape[0], device=input.device)
    if share_lam:
        lam = beta.sample().to(device=input.device)
        lam = torch.max(lam, 1. - lam)
        lam_expanded = lam
    else:
        lam = beta.sample([input.shape[0]]).to(device=input.device)
        lam = torch.max(lam, 1. - lam)
        lam_expanded = lam.view([-1] + [1]*(input.dim()-1))
    output = lam_expanded * input + (1. - lam_expanded) * input[randind]
    return output, randind, lam



# SEED
def set_seed(seed):
    if_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    if if_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_log(print_str, log_file, print_=False):
    if print_:
        print(print_str)
    if log_file is None:
        return
    with open(log_file, 'a') as f:
        f.write('\n')
        f.write(print_str)


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


# test
if __name__ == '__main__':
    logits = torch.rand(2048, 2048).cuda()
    bsz = logits.shape[0]
    labels = torch.arange(bsz, dtype=torch.long).cuda()
    lam = torch.rand(2048).cuda()
    weights = torch.rand(2048).cuda()

    sample_dim, label_dim = logits.size(0), logits.size(1)
    logits_exp = logits.exp()
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(logits.device)
    weights = weights.unsqueeze(0).repeat(sample_dim, 1)
    weight_pos = weights[mask]
    weights_mask = weights * (~mask)

    weight_neg_norm = weights_mask / weights_mask.sum(1).unsqueeze(1) * (label_dim-1)
    weights_new = mask + weight_neg_norm

    softmax_loss = (weights_new*logits_exp) / (weights_new*logits_exp).sum(1).unsqueeze(1)
    cont_loss_env = lam * torch.nn.NLLLoss(reduction='none')(torch.log(softmax_loss), labels)
    cont_loss_env = (cont_loss_env * weight_pos.squeeze()).sum() / weight_pos.sum()
    print(cont_loss_env)