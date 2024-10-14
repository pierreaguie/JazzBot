from JazzBot.data_processor import *
from torch.utils.data import DataLoader, Dataset, random_split
import torch


class MyDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.input_data[idx], dtype=torch.long)
        output_tensor = torch.tensor(self.output_data[idx], dtype=torch.long)
        return input_tensor, output_tensor
    

def dataloader(folder_path, batch_size, N):
    '''
    Args : N = number of notes we want in the pieces
    '''
    input, target = folderToVectInputTarget(folder_path,N)
    dataset = MyDataset(input, target)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_val_dataloaders(input, target, batch_size):
    dataset = MyDataset(input, target)
    # use 20% of training data for validation
    n = len(input)
    train_set_size = int(n * 0.8)
    valid_set_size = n - train_set_size
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size], generator=seed)
    
    # Create a dataloader
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_dataloader, val_dataloader

