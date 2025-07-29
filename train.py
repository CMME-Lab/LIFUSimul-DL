import os
import time
import argparse
import torch
import pickle
from torch.optim.lr_scheduler import LambdaLR
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')

from dataset import create_dataloader
from models.cnn import CNNs
from models.swin import SwinUNet
from utils import train_one_epoch, val_one_epoch
from config import load_train_config

if __name__ == "__main__":
    
    opt = load_train_config()

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - opt.num_epoch) / float(opt.decay_epoch)
        return lr_l

    os.makedirs(os.path.join('runs', opt.run_name), exist_ok=True)
    
    device = torch.device('cuda') if opt.cuda else 'cpu'

    # Define dataset
    train_dataloader, valid_dataloader = create_dataloader(opt.data_path, opt.modality, opt.num_subjects, 
                                                           opt.num_data, opt.train_bs, opt.valid_bs, opt.valid_ratio,
                                                           test=False, test_bs=None)
    train_epoch = opt.num_epoch + opt.decay_epoch

    # Define model: load your saved state dict if needed
    if opt.model == 'ae': model = CNNs(in_channels=opt.in_channels, out_channels=opt.out_channels,
                                       channels=opt.channel_dims, skip=False, data_size=(opt.d1, opt.d2, opt.d3))

    elif opt.model=='unet': model = CNNs(in_channels=opt.in_channels, out_channels=opt.out_channels,
                                         channels=opt.channel_dims, skip=True, data_size=(opt.d1, opt.d2, opt.d3))

    elif opt.model == 'swin': model = SwinUNet(in_channels=opt.in_channels, out_channels=opt.out_channels,
                                               in_dim=opt.channel_dims, device=device, data_size=(opt.d1, opt.d2, opt.d3))
    
    if opt.init_model:
        model.apply(weights_init_swin) if opt.model=='swin' else model.apply(weights_init_cnn)
    
    model = model.to(device)
    
    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda = lambda_rule) if opt.decay_epoch!=0 else None

    val_score_max = 0
    train_results = defaultdict(list)
    
    # Training process
    for epoch in range(1, train_epoch+1):
        
        start_time = time.time()

        train_metrics = train_one_epoch(epoch, train_epoch, model, optimizer, train_dataloader, device, scheduler)
        valid_metrics = val_one_epoch(epoch, train_epoch, model, valid_dataloader, device)

        for key, value in train_metrics.items():
            train_results[f'train_{key}'].append(value)

        for key, value in valid_metrics.items():
            train_results[f'valid_{key}'].append(value)
        
        if train_results['valid_dice'][-1] > val_score_max:
            val_score_max = train_results['valid_dice'][-1]
            torch.save(model.state_dict(), f"runs/{opt.run_name}/best_model.pth")
    
    # save the trained model and results
    torch.save(model.state_dict(), f"runs/{opt.run_name}/epoch_{train_epoch}.pth")
    
    with open(f"runs/{opt.run_name}/train_results.pth", 'wb') as f:
        pickle.dump(train_results, f)