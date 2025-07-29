# helpers
import os
import sys
import torch
from datetime import timedelta
from time import time
import numpy as np
import math
import matplotlib.pyplot as plt

def fourier_feature_embed(x, out_dim):
    B = torch.randn(x.shape[1], out_dim, requires_grad=False).to(x.device)
    x = (2.0 * pi * x @ B)

    return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

class MinMaxScaling:
    '''
    input : pytorch tensor
    perform Min-Max scaling to batched input (expected size B*1*D*H*W)
    '''
    def __init__(self, X):
        self.min_value = X.view(X.size(0),-1).min(1)[0]
        self.max_value = X.view(X.size(0),-1).max(1)[0]
        
        self.ff_max_value = 479.1307 # WARNING: This was set manually with our own data.
        self.ff_min_value = 1.4112
    
    def fit_transform(self, X, ff=True):
        X_std = X.clone()
        if ff:
            for i in range(len(X_std)):
                X_std[i] = (X[i] - self.ff_min_value) / (self.ff_max_value - self.ff_min_value)    
        
        else:
            for i in range(len(X_std)):
                X_std[i] = (X[i] - self.min_value[i]) / (self.max_value[i] - self.min_value[i])    
        
        return X_std
    
    def inverse_transform(self, X, ff=True):
        X_inv = X.clone()
        
        if ff:
            for i in range(len(X_inv)):
                X_inv[i] = X[i] * (self.ff_max_value - self.ff_min_value) + self.ff_min_value
            
        else:
            for i in range(len(X_inv)):
                X_inv[i] = X[i] * (self.max_value[i] - self.min_value[i]) + self.min_value[i]
        
        return X_inv

def thr_npy(npy):
    '''
    input : numpy array
    for thresholding numpy arrays
    '''
    max_value = npy.max()*0.5
    return (npy>=max_value).astype(float)
        
def calculate_dice(pred, target):
    '''
    input : pytorch tensor
    computes dice score between two tensors
    '''
    max_pred = torch.max(pred)
    max_target = torch.max(target)
    
    pred = (pred.contiguous().view(-1)>=max_pred*0.5).float()
    target = (target.contiguous().view(-1)>=max_target*0.5).float()
    
    count_p = torch.count_nonzero(pred)
    count_t = torch.count_nonzero(target)
    count = torch.count_nonzero(((pred==1)&(target==1)).float())

    dice = count*2 / (count_p+count_t)
    
    return dice


def calculate_ep(pred, target):
    max_pred = torch.max(pred)
    max_target = torch.max(target)
    ep = abs(max_pred - max_target)
    
    return ep


def batch_metrics(pred_list, target_list):
    total_dice = 0
    total_ep = 0
    batch_size = pred_list.size(0)
    
    for i in range(batch_size):
        pred = pred_list[i].detach()
        target = target_list[i].detach()
        
        dice = calculate_dice(pred, target)
        ep = calculate_ep(pred, target)
        
        total_dice += dice
        total_ep += ep
        
    return total_dice/batch_size, total_ep/batch_size


def train_one_epoch(epoch, train_epoch, model, optimizer, train_dataloader, device, scheduler=None):

    model.train()
    
    loss_tot = 0
    dice_tot = 0
    ep_tot = 0

    criterion = torch.nn.MSELoss().to(device)
    total_batches = train_epoch * (len(train_dataloader))
    prev_time = time()
    
    for i, batch in enumerate(train_dataloader):
        
        # Model inputs
        ff = batch["FF"].unsqueeze(1).to(device)
        skull = batch["SK"].unsqueeze(1).to(device)
        tinput = batch["TD"].to(device)
        target = batch["Y"].unsqueeze(1).to(device)
        
        scaler = MinMaxScaling(ff)
        ff = scaler.fit_transform(ff, ff=False)
        target = scaler.fit_transform(target, ff=True)

        optimizer.zero_grad()

        # compute loss
        pred = model(ff, skull, tinput)
        
        loss = criterion(pred, target)
        loss.mean().backward()
        loss_tot += loss.mean().item()     
        
        optimizer.step()
        
        # Calculate dice score
        train_dice, train_ep = batch_metrics(pred, target)
        dice_tot += train_dice
        ep_tot += train_ep
        
        batches_done = (epoch - 1) * (len(train_dataloader)) + i + 1
        batches_left = total_batches - batches_done
        
        time_remain = timedelta(seconds=batches_left * (time() - prev_time))
        prev_time = time()

        sys.stdout.write(
            "\r[Epoch %d/%d] [Train batch %d/%d] [loss : %f] [Dice : %.2f] [Ep : %.2f] [Train ETA : %s]"
            % (
                epoch,
                train_epoch,
                i+1,
                len(train_dataloader),
                loss.mean().item(),
                train_dice*100,
                train_ep*100,
                str(time_remain)[:-7]
            )
        )
        sys.stdout.flush()
        print('')
    
    if scheduler!=None: scheduler.step()

    return {'loss':loss_tot/len(train_dataloader), 'dice':dice_tot/len(train_dataloader), 'Ep':ep_tot/len(train_dataloader)}
    
def val_one_epoch(epoch, train_epoch, model, valid_dataloader, device):

    model.eval()
    loss_tot = 0
    dice_tot = 0
    ep_tot = 0
    criterion = torch.nn.MSELoss().to(device)
    
    total_batches = train_epoch * (len(valid_dataloader))
    prev_time = time()

    with torch.no_grad():
        
        for i, batch in enumerate(valid_dataloader):
            
            # Model inputs
            ff = batch["FF"].unsqueeze(1).to(device)
            skull = batch["SK"].unsqueeze(1).to(device)
            tinput = batch["TD"].to(device)
            target = batch["Y"].unsqueeze(1).to(device)

            scaler = MinMaxScaling(ff)
            ff = scaler.fit_transform(ff, ff=False)
            target = scaler.fit_transform(target, ff=True)

            # compute loss
            pred = model(ff, skull, tinput)
            loss = criterion(pred, target)
            loss_tot += loss.mean().item()     
            
            # Calculate dice score
            valid_dice, valid_ep = batch_metrics(pred, target)
            dice_tot += valid_dice
            ep_tot += valid_ep
            
            batches_done = (epoch - 1) * (len(valid_dataloader)) + i + 1
            batches_left = total_batches - batches_done
            
            time_remain = timedelta(seconds=batches_left * (time() - prev_time))
            prev_time = time()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Valid batch %d/%d] [loss : %f] [Dice : %.2f] [Ep : %.2f] [Val ETA : %s]"
                % (
                    epoch,
                    train_epoch,
                    i+1,
                    len(valid_dataloader),
                    loss.mean().item(),
                    valid_dice*100,
                    valid_ep*100,
                    str(time_remain)[:-7]
                )
            )
            sys.stdout.flush()
    
    print('')
        
    return {'loss':loss_tot/len(valid_dataloader), 'dice':dice_tot/len(valid_dataloader), 'Ep':ep_tot/len(valid_dataloader)}

def test_loop(model, test_dataloader, device):

    pred_list = []
    target_list = []
    
    total_dice = []
    total_ep = []

    time_list = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):

            start = time()

            # Model inputs
            ff = batch["FF"].unsqueeze(1).to(device)
            skull = batch["SK"].unsqueeze(1).to(device)
            tinput = batch["TD"].to(device)
            target = batch["Y"].unsqueeze(1)
            
            # Optional : apply MinMaxscaling to inputs
            scaler = MinMaxScaling(ff)
            ff = scaler.fit_transform(ff, ff=False)
            target = scaler.fit_transform(target, ff=True)

            # output prediction
            pred = model(ff, skull, tinput).cpu()
            dice, ep = batch_metrics(pred, target)

            total_dice.append(dice)
            total_ep.append(ep)
            
            pred_list.append(pred.squeeze().detach())
            target_list.append(target.squeeze())

            sys.stdout.write(
                f"\r[Batch {i+1}/{len(test_dataloader)}] processing..."
            )
            sys.stdout.flush()

            time_list.append((time() - start) / pred.shape[0])

    # flattening result list
    target_list = [item for sublist in target_list for item in sublist]
    pred_list = [item for sublist in pred_list for item in sublist]
    time_per_data = sum(time_list) / len(time_list)

    return target_list, pred_list, total_dice, total_ep, time_per_data
    
def savefig(results, run_name, thsd=False):

    if thsd:
        path = os.path.join("runs", run_name, "images/thsd")
        name = "thsd"
        cmap = "viridis"
    else:
        path = os.path.join("runs", run_name, "images")
        name = "pmap"
        cmap = "jet"
        
    os.makedirs(path, exist_ok=True)

    s1 = int(results[0].shape[0]/2)
    s2 = int(results[0].shape[1]/2)
    s3 = int(results[0].shape[2]/2)
    
    for i in range(len(results)):
        output = np.array(results[i])
        if thsd: output = thr_npy(output)

        plt.figure(figsize=(15, 5.3))

        plt.subplot(1,3,1)
        plt.title('Axis 0')
        plt.imshow(output[s1,:,:], cmap=cmap)
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.title('Axis 1')
        plt.imshow(output[:,s2,:], cmap=cmap)
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.title('Axis 2')
        plt.imshow(output[:,:,s3], cmap=cmap)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(path, f"{name}_{i+1}.png"), bbox_inches='tight')
        plt.close()

        sys.stdout.write(
            f'\rSaving plots {name} [{i+1}/{len(results)}]'
        )
        sys.stdout.flush()