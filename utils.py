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

np.random.seed(0)


class STL10Pair(STL10):
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target


class STL10Pair_Index(STL10):
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target, index


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

class CIFAR10Pair_Index(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target, index

class CIFAR100Pair(CIFAR100):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

class CIFAR100Pair_Index(CIFAR100):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target, index

def group_crossentropy(logits, labels, batchsize):
    sample_dim, label_dim = logits.size(0), logits.size(1)
    logits_exp = logits.exp()
    weights = torch.ones_like(logits_exp)
    weights[:, 1:] *= (batchsize-2)/(label_dim-1)
    softmax_loss = (weights * logits_exp) / (weights * logits_exp).sum(1).unsqueeze(1)
    cont_loss_env = torch.nn.NLLLoss()(torch.log(softmax_loss), labels)
    return cont_loss_env


def info_nce_loss(features, batch_size, temperature):

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    # features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    logits = logits / temperature
    return logits, labels


def info_nce_loss_update(features, batch_size, temperature):

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)
    # split_matrixs = torch.cat([split_matrix, split_matrix], dim=0).to(features.device)
    index_sequence = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0).to(features.device)
    index_sequence = index_sequence.unsqueeze(0).expand(2*batch_size, 2*batch_size)

    # features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # split_matrixs = split_matrixs[~mask].view(split_matrixs.shape[0], -1)
    index_sequence = index_sequence[~mask].view(index_sequence.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    positive_index = index_sequence[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    negative_index = index_sequence[~labels.bool()].view(labels.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)
    indexs = torch.cat([positive_index, negative_index], dim=1)

    logits = logits / temperature
    return logits, labels, indexs


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


# Update split online
def auto_split(net, update_loader, soft_split_all, temperature, irm_temp, loss_mode='v2', irm_mode='v1', irm_weight=10, constrain=False, cons_relax=False, nonorm=False, log_file=None):
    # irm mode: v1 is original irm; v2 is variance (not use)

    low_loss, constrain_loss = 1e5, torch.Tensor([0.])
    cnt, best_epoch, training_num = 0, 0, 0
    num_env = soft_split_all.size(1)

    # optimizer and schedule
    pre_optimizer = torch.optim.Adam([soft_split_all], lr=0.5, weight_decay=0.)
    pre_scheduler = MultiStepLR(pre_optimizer, [5, 35], gamma=0.2, last_epoch=-1)

    for epoch in range(100):
        risk_all_list, risk_cont_all_list, risk_penalty_all_list, risk_constrain_all_list, training_num = [],[],[],[], 0
        net.eval()
        for batch_idx, (pos_1, pos_2, target, idx) in enumerate(update_loader):
            training_num += len(pos_1)
            with torch.no_grad():
                _, feature_1 = net(pos_1.cuda(non_blocking=True))
                _, feature_2 = net(pos_2.cuda(non_blocking=True))

            loss_cont_list, loss_penalty_list = [], []

            '''
            # Option 1. use probability directly
            soft_split = F.softmax(soft_split_all, dim=-1)
            for env_idx in range(num_env):
                loss_weight = torch.gather(soft_split[:, env_idx], dim=1, index=indexs)
                cont_loss_env_sample = (loss_weight*loss_original).sum(1)
                cont_loss_env = (cont_loss_env_sample * torch.cat([soft_split[:, env_idx], soft_split[:, env_idx]], dim=0)).sum(0)
                loss_cont_list.append(cont_loss_env)
                penalty_irm = torch.autograd.grad(cont_loss_env, [scale], create_graph=True)[0]
                loss_penalty_list.append(penalty_irm)
            risk_final = - (loss_cont_list.sum() + loss_penalty_list.sum())
            '''

            # Option 2. use soft split
            param_split = F.softmax(soft_split_all[idx], dim=-1)
            if irm_mode == 'v1': # original
                for env_idx in range(num_env):

                    logits, labels, indexs = info_nce_loss_update(torch.cat([feature_1, feature_2], dim=0), feature_1.size(0), temperature=1.0)

                    loss_weight = param_split[:, env_idx][indexs]
                    logits_cont = logits / temperature
                    # here we change the contrastive loss to the soft version to enable the sample weight
                    cont_loss_env = soft_contrastive_loss(logits_cont, labels, loss_weight, mode=loss_mode, nonorm=nonorm)

                    scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
                    logits_pen = logits / irm_temp
                    cont_loss_env_scale = soft_contrastive_loss(logits_pen*scale, labels, loss_weight, mode=loss_mode, nonorm=nonorm)
                    penalty_irm = torch.autograd.grad(cont_loss_env_scale, [scale], create_graph=True)[0]
                    loss_cont_list.append(cont_loss_env)
                    loss_penalty_list.append(torch.sum(penalty_irm**2))

                cont_loss_epoch = torch.stack(loss_cont_list).mean()
                inv_loss_epoch = torch.stack(loss_penalty_list).mean()
                risk_final = - (cont_loss_epoch + irm_weight*inv_loss_epoch)


            elif irm_mode == 'v2': # variance (not use)
                for env_idx in range(num_env):
                    logits, labels, indexs = info_nce_loss_update(torch.cat([feature_1, feature_2], dim=0), feature_1.size(0), temperature=1.0)
                    loss_weight = param_split[:, env_idx][indexs]
                    logits_cont = logits / temperature
                    cont_loss_env = soft_contrastive_loss(logits_cont, labels, loss_weight, mode=loss_mode, nonorm=nonorm)
                    loss_cont_list.append(cont_loss_env)

                inv_loss_epoch = torch.var(torch.stack(loss_cont_list))
                cont_loss_epoch = torch.stack(loss_cont_list).mean()
                risk_final = - (cont_loss_epoch + irm_weight*inv_loss_epoch)

            if constrain: # constrain to avoid the imbalance problem
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
                print('\rUpdating Env [%d/%d] [%d/%d]  Loss: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2f  Cons_Risk: %.2f  Cnt: %d  Lr: %.4f  Inv_Mode: %s  Soft Split: %s'
                      %(epoch, 100, training_num, len(update_loader.dataset), sum(risk_all_list)/len(risk_all_list), sum(risk_cont_all_list)/len(risk_cont_all_list), sum(risk_penalty_all_list)/len(risk_penalty_all_list),
                        sum(risk_constrain_all_list)/len(risk_constrain_all_list), cnt, pre_optimizer.param_groups[0]['lr'], irm_mode, F.softmax(soft_split_print, dim=-1)), end='', flush=True)


        pre_scheduler.step()
        avg_risk = sum(risk_all_list)/len(risk_all_list)
        avg_cont_risk = sum(risk_cont_all_list)/len(risk_cont_all_list)
        avg_inv_risk = sum(risk_penalty_all_list)/len(risk_penalty_all_list)

        if epoch == 0:
            write_log("Initial Risk: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2f" %(avg_risk, avg_cont_risk, avg_inv_risk), log_file=log_file, print_=True)
            soft_split_best = soft_split_all.clone().detach()
        if avg_risk < low_loss:
            low_loss = avg_risk
            soft_split_best = soft_split_all.clone().detach()
            best_epoch = epoch
            cnt = 0
        else:
            cnt += 1

        if epoch > 50 and cnt >= 5 or epoch == 60:
            write_log('\nLoss not down. Break down training.  Epoch: %d  Loss: %.2f' %(best_epoch, low_loss), log_file=log_file, print_=True)
            write_log('Updating Env [%d/%d] [%d/%d]  Loss: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2f  Cons_Risk: %.2f  Cnt: %d  Lr: %.4f  Inv_Mode: %s'
                      %(epoch, 100, training_num, len(update_loader.dataset), sum(risk_all_list)/len(risk_all_list), sum(risk_cont_all_list)/len(risk_cont_all_list), sum(risk_penalty_all_list)/len(risk_penalty_all_list),
                        sum(risk_constrain_all_list)/len(risk_constrain_all_list), cnt, pre_optimizer.param_groups[0]['lr'], irm_mode), log_file=log_file)
            final_split_softmax = F.softmax(soft_split_best, dim=-1)
            write_log('%s' %(final_split_softmax), log_file=log_file, print_=True)
            group_assign = final_split_softmax.argmax(dim=1)
            write_log('Debug:  group1 %d  group2 %d' %(group_assign.sum(), group_assign.size(0)-group_assign.sum()), log_file=log_file, print_=True)
            return soft_split_best


# update split offline
def auto_split_offline(out_1, out_2, soft_split_all, temperature, irm_temp, loss_mode='v2', irm_mode='v1', irm_weight=10, constrain=False, cons_relax=False, nonorm=False, log_file=None):
    # irm mode: v1 is original irm; v2 is variance
    low_loss, constrain_loss = 1e5, torch.Tensor([0.])
    cnt, best_epoch, training_num = 0, 0, 0
    num_env = soft_split_all.size(1)
    # optimizer and schedule
    pre_optimizer = torch.optim.Adam([soft_split_all], lr=0.5, weight_decay=0.)
    pre_scheduler = MultiStepLR(pre_optimizer, [5, 35], gamma=0.2, last_epoch=-1)

    # dataset and dataloader
    traindataset = update_split_dataset(out_1, out_2)
    trainloader = DataLoader(traindataset, batch_size=3096, shuffle=True, num_workers=4)

    for epoch in range(100):
        risk_all_list, risk_cont_all_list, risk_penalty_all_list, risk_constrain_all_list, training_num = [],[],[],[], 0

        for feature_1, feature_2, idx in trainloader:
            feature_1, feature_2 = feature_1.cuda(), feature_2.cuda()
            loss_cont_list, loss_penalty_list = [], []
            training_num += len(feature_1)

            param_split = F.softmax(soft_split_all[idx], dim=-1)
            if irm_mode == 'v1': # original
                for env_idx in range(num_env):
                    logits, labels, indexs = info_nce_loss_update(torch.cat([feature_1, feature_2], dim=0), feature_1.size(0), temperature=1.0)

                    loss_weight = param_split[:, env_idx][indexs]
                    logits_cont = logits / temperature
                    cont_loss_env = soft_contrastive_loss(logits_cont, labels, loss_weight, mode=loss_mode, nonorm=nonorm)

                    scale = torch.ones((1, logits.size(-1))).cuda().requires_grad_()
                    logits_pen = logits / irm_temp
                    cont_loss_env_scale = soft_contrastive_loss(logits_pen*scale, labels, loss_weight, mode=loss_mode, nonorm=nonorm)
                    penalty_irm = torch.autograd.grad(cont_loss_env_scale, [scale], create_graph=True)[0]
                    loss_cont_list.append(cont_loss_env)
                    loss_penalty_list.append(torch.sum(penalty_irm**2))

                cont_loss_epoch = torch.stack(loss_cont_list).mean()
                inv_loss_epoch = torch.stack(loss_penalty_list).mean()
                risk_final = - (cont_loss_epoch + irm_weight*inv_loss_epoch)


            elif irm_mode == 'v2': # variance (not use)
                for env_idx in range(num_env):
                    logits, labels, indexs = info_nce_loss_update(torch.cat([feature_1, feature_2], dim=0), feature_1.size(0), temperature=1.0)
                    loss_weight = param_split[:, env_idx][indexs]
                    logits_cont = logits / temperature
                    cont_loss_env = soft_contrastive_loss(logits_cont, labels, loss_weight, mode=loss_mode, nonorm=nonorm)
                    loss_cont_list.append(cont_loss_env)

                inv_loss_epoch = torch.var(torch.stack(loss_cont_list))
                cont_loss_epoch = torch.stack(loss_cont_list).mean()
                risk_final = - (cont_loss_epoch + irm_weight*inv_loss_epoch)

            if constrain: # constrain to avoid the imbalance problem
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
                print('\rUpdating Env [%d/%d] [%d/%d]  Loss: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2f  Cons_Risk: %.2f  Cnt: %d  Lr: %.4f  Inv_Mode: %s  Soft Split: %s'
                      %(epoch, 100, training_num, len(trainloader.dataset), sum(risk_all_list)/len(risk_all_list), sum(risk_cont_all_list)/len(risk_cont_all_list), sum(risk_penalty_all_list)/len(risk_penalty_all_list),
                        sum(risk_constrain_all_list)/len(risk_constrain_all_list), cnt, pre_optimizer.param_groups[0]['lr'], irm_mode, F.softmax(soft_split_print, dim=-1)), end='', flush=True)

        pre_scheduler.step()
        avg_risk = sum(risk_all_list)/len(risk_all_list)
        avg_cont_risk = sum(risk_cont_all_list)/len(risk_cont_all_list)
        avg_inv_risk = sum(risk_penalty_all_list)/len(risk_penalty_all_list)

        if epoch == 0:
            write_log("Initial Risk: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2f" % (avg_risk, avg_cont_risk, avg_inv_risk), log_file=log_file, print_=True)
            soft_split_best = soft_split_all.clone().detach()
        if avg_risk < low_loss:
            low_loss = avg_risk
            soft_split_best = soft_split_all.clone().detach()
            best_epoch = epoch
            cnt = 0
        else:
            cnt += 1

        if epoch > 50 and cnt >= 5 or epoch == 60:
            write_log('\nLoss not down. Break down training.  Epoch: %d  Loss: %.2f' %(best_epoch, low_loss), log_file=log_file, print_=True)
            write_log('Updating Env [%d/%d] [%d/%d]  Loss: %.2f  Cont_Risk: %.2f  Inv_Risk: %.2f  Cons_Risk: %.2f  Cnt: %d  Lr: %.4f  Inv_Mode: %s'
                      %(epoch, 100, training_num, len(trainloader.dataset), sum(risk_all_list)/len(risk_all_list), sum(risk_cont_all_list)/len(risk_cont_all_list), sum(risk_penalty_all_list)/len(risk_penalty_all_list),
                        sum(risk_constrain_all_list)/len(risk_constrain_all_list), cnt, pre_optimizer.param_groups[0]['lr'], irm_mode), log_file=log_file)
            final_split_softmax = F.softmax(soft_split_best, dim=-1)
            write_log('%s' %(final_split_softmax), log_file=log_file, print_=True)
            group_assign = final_split_softmax.argmax(dim=1)
            write_log('Debug:  group1 %d  group2 %d' %(group_assign.sum(), group_assign.size(0)-group_assign.sum()), log_file=log_file, print_=True)
            return soft_split_best


# soft version of the contrastive loss
def soft_contrastive_loss(logits, labels, weights, mode='v1', nonorm=False):
    if mode == 'v1':
        logits *= weights
        cont_loss_env = torch.nn.CrossEntropyLoss()(logits, labels)
    elif mode == 'v2':
        sample_dim, label_dim = logits.size(0), logits.size(1)
        logits_exp = logits.exp()
        weight_pos, weight_neg = torch.split(weights, [1, label_dim-1], dim=1)
        weight_neg_norm = weight_neg / weight_neg.sum(1).unsqueeze(1) * (label_dim-1)
        weights_new = torch.cat([torch.ones_like(weight_pos), weight_neg_norm], dim=1)
        softmax_loss = (weights_new*logits_exp) / (weights_new*logits_exp).sum(1).unsqueeze(1)
        cont_loss_env = torch.nn.NLLLoss(reduction='none')(torch.log(softmax_loss), labels)
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

# just follow the previous work -- DCL, NeurIPS2020
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=int(0.1 * 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
