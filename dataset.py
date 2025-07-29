import os
import h5py
import torch
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader, Subset

class LIFUData(Dataset):

    def __init__(self, data_path, split, modality):

        self.path = data_path        # Where data is located
        self.split = split      # train / test

        # Multi-Modal Inputs
        self.ff = self.load_hdf5("ff", self.split) # Acoustic Free-Field  
        self.sk = self.load_hdf5(modality, self.split) # Skull Images (CT / MR)
        self.td = self.load_hdf5("td", self.split) # Transducer Placements

        # Output
        self.Y = self.load_hdf5("target", self.split) # Intracranial Acoustic Field

        # Data index
        self.idx = natsorted(list(self.ff.keys()))

    def __len__(self):

        return len(self.idx)

    def __getitem__(self, idx):

        i = self.idx[idx]
        
        ff = torch.Tensor(self.ff[i][()]).float()
        sk = torch.Tensor(self.sk[i][()]).float()
        td = torch.Tensor(self.td[i][()]).float()
        y = torch.Tensor(self.Y[i][()]).float()

        return {"FF":ff, "SK":sk, "TD":td, "Y":y}

    def load_hdf5(self, modality, split):

        path = os.path.join(self.path, f"{modality}_{split}.hdf5")
        X = h5py.File(path, 'r')

        return X

def split_dataset(num_subjects, num_data, valid_ratio):
    data_length = num_subjects * num_data
    sub_length = int(num_data * valid_ratio)

    test_indices = []

    for i in range(num_subjects):
        start_idx = i * (data_length // num_subjects)
        subject_indices = torch.arange(start_idx, start_idx + (data_length // num_subjects))[:sub_length]
        test_indices.append(subject_indices)

    test_indices = torch.cat(test_indices)
    train_indices = torch.tensor([idx for idx in range(data_length) if idx not in test_indices])

    return train_indices, test_indices

def create_dataloader(data_path, modality, num_subjects, num_data, train_bs, valid_bs, valid_ratio, test=False, test_bs=None):

    if test:
        test_set = LIFUData(data_path, split='test', modality=modality)
        test_dataloader = DataLoader(
            test_set,
            batch_size = test_bs,
            shuffle = False
        )

        return test_dataloader

    train_set = LIFUData(data_path, split='train', modality=modality)
    train_indices, valid_indices = split_dataset(num_subjects, num_data, valid_ratio)

    train_data = Subset(train_set, train_indices)
    valid_data = Subset(train_set, valid_indices)

    train_dataloader = DataLoader(
        train_data,
        batch_size = train_bs,
        shuffle = True
    )

    valid_dataloader = DataLoader(
        valid_data,
        batch_size = valid_bs,
        shuffle = False
    )

    return train_dataloader, valid_dataloader