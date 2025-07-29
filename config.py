import ast
import argparse

def load_base_config():
    parser = argparse.ArgumentParser(add_help = False)

    parser.add_argument('--data_path', type=str, default='.', help="Path where dataset is located")
    parser.add_argument('--run_name', type=str, default='test_run', help="Name of current experiment")
    parser.add_argument('--modality', type=str, default='mr', help="Skull image modality (CT/MR)")
    parser.add_argument('--model', type=str, default='unet', help="Choose Model to use [ae/unet/swin]")
    parser.add_argument('--cuda', action='store_true', default=False, help="Turn on when using GPU devices")
    parser.add_argument('--num_subjects', type=int, default=8, help="Number of subjects (skull) of data")
    parser.add_argument('--num_data', type=int, default=400, help="Number of data per each subject")
    parser.add_argument('--d1', type=int, default=112, help="First dimension of your data : (d1, d2, d3)")
    parser.add_argument('--d2', type=int, default=112, help="Second dimension of your data : (d1, d2, d3)")
    parser.add_argument('--d3', type=int, default=112, help="Third dimension of your data : (d1, d2, d3)")
    parser.add_argument('--in_channels', type=int, default=1, help="Input channel dimension of your data")
    parser.add_argument('--out_channels', type=int, default=1, help="Output channel dimension of your data")
    parser.add_argument('--channel_dims', type=int, default=32, help="Channel or dimension to expand (recommend: 32 in CNNs, 96 in Swin)")
    
    return parser

def load_train_config():
    base_parser = load_base_config()

    parser = argparse.ArgumentParser(parents=[base_parser])

    parser.add_argument("--num_epoch", type=int, default=100, help="train epoch for constant lr")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch for lr decay, set to 0 if not using learning rate scheduler")
    parser.add_argument("--train_bs", type=int, default=8, help="batch size for training")
    parser.add_argument("--valid_bs", type=int, default=8, help="batch size for validation")
    parser.add_argument("--valid_ratio", type=float, default=0.2, help="train / valid split ratio")
    parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="Adam coefficient")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam coefficient")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Adam regularization")
    parser.add_argument("--init_model", action='store_true', default=False, help="Turn on when training from scratch")

    return parser.parse_args()

def load_test_config():
    base_parser = load_base_config()

    parser = argparse.ArgumentParser(parents=[base_parser])

    parser.add_argument('--test_bs', type=int, default=8, help="Test batch size")
    parser.add_argument('--plot', action='store_true', default=False, help="Save result images")

    return parser.parse_args()