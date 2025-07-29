import os
import argparse
import numpy as np
import torch

import warnings
warnings.filterwarnings("ignore")

from dataset import create_dataloader
from models.cnn import CNNs
from models.swin import SwinUNet
from utils import test_loop, savefig
from config import load_test_config


if __name__ == "__main__":

    opt = load_test_config()

    path = os.path.join("runs", opt.run_name, "best_model.pth")

    test_dataloader = create_dataloader(opt.data_path, opt.modality, opt.num_subjects,
                                        opt.num_data, train_bs=None, valid_bs=None, valid_ratio=None,
                                        test=True, test_bs=opt.test_bs)
    device = torch.device('cuda') if opt.cuda else 'cpu'

    # Define model
    if opt.model == 'ae': model = CNNs(in_channels=opt.in_channels, out_channels=opt.out_channels,
                                       channels=opt.channel_dims, skip=False, data_size=(opt.d1, opt.d2, opt.d3))

    elif opt.model=='unet': model = CNNs(in_channels=opt.in_channels, out_channels=opt.out_channels,
                                         channels=opt.channel_dims, skip=True, data_size=(opt.d1, opt.d2, opt.d3))

    elif opt.model == 'swin': model = SwinUNet(in_channels=opt.in_channels, out_channels=opt.out_channels,
                                               in_dim=opt.channel_dims, device=device, data_size=(opt.d1, opt.d2, opt.d3))
    
    saved_dict = torch.load(path, map_location = device)
    model.load_state_dict(saved_dict['model_state_dict']) if 'model_state_dict' in saved_dict.keys() else model.load_state_dict(saved_dict)
    model.to(device)

    target, pred, dice, ep, eval_time = test_loop(model, test_dataloader, device)
    
    print("\n===============================")
    print("Dice : %.2f +- %.2f"%(np.mean(dice)*100, np.std(dice)*100))
    print("Ep : %.2f +- %.2f"%(np.mean(ep)*100, np.std(ep)*100))
    print('Inference time : %.6f'%eval_time)

    eval_results = {'dice':dice, 'ep':ep, 'time':eval_time}
    with open(os.path.join('runs',opt.run_name,'eval_results.pickle'),'wb') as f:
        pickle.dump(eval_results, f)
    
    # optional : save result images
    if opt.plot:
        savefig(target, opt.run_name, thsd=False)
        print('')
        savefig(target, opt.run_name, thsd=True)
