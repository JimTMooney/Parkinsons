from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch

def loader_from_idxs(dataset, idxs, collate_fn, batch_size):
    sampler = SubsetRandomSampler(idxs)
    return DataLoader(dataset, sampler=sampler, batch_size = batch_size,
                      num_workers=0, collate_fn = collate_fn)

def get_loaders(dataset, fold=0, collate_fn=None, batch_size = 16):
    train_idxs, test_idxs = dataset.split_data(fold)
    
    train_loader = loader_from_idxs(dataset, train_idxs, collate_fn, batch_size)
    test_loader = loader_from_idxs(dataset, test_idxs, collate_fn, batch_size)

    return train_loader, test_loader



def mask_collate_fn(device):
    def mask_helper(data):
        signals, labels = zip(*data)
        signals = list(signals)
        max_sz = -1
        for signal in signals:
            sig_sz = signal.shape[2]
            if sig_sz > max_sz:
                max_sz = sig_sz

        for idx in range(len(signals)):
            d, h, w = signals[idx].shape
            if w < max_sz:
                zero_pad = torch.zeros((d, h, max_sz - w)).to(device)
                signals[idx] = torch.cat((signals[idx], zero_pad), 2)

        return torch.stack(signals), torch.tensor(labels).to(device)
    return mask_helper
    

def single_collate_fn(device):
    def single_helper(data):
        signals, labels = zip(*data)
        signals = list(signals)
        labels = list(labels)
        for idx, sig in enumerate(signals):
            signals[idx] = torch.unsqueeze(sig, 0)
        return signals, torch.unsqueeze(torch.tensor(labels), 1).to(device)
    return single_helper