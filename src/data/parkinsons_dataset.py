import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import math


class ParkinsonsDataset(Dataset):
    def __init__(self):
        self.device = None
        self.root_dir = None

        self.files_by_class = []
        self.n_data = 0

        self.train_mode = True

        self.data = {}
        self.fold_idxs = []

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        raise NotImplementedError("Implement the __getitem__ function from ParkinsonsDataset class")

    def cv_init(self, n_cv=5):
        file_ids = []
        for class_idx, class_files in enumerate(self.files_by_class):
            file_ids.append([None]*len(class_files))
            for f_idx, filename in enumerate(class_files):
                file_ids[class_idx][f_idx] = self.id_filter(filename)

        self.id_sets = [None] * n_cv
        cv_data = [None] * n_cv
        for cv_idx in range(n_cv):
            cv_data[cv_idx] = []
            self.id_sets[cv_idx] = set()
            

        for class_idx, class_files in enumerate(self.files_by_class):
            ids = np.unique(file_ids[class_idx])
            id_mapper = dict(zip(ids, np.arange(len(ids))))
            permute_arr = np.random.permutation(len(ids))
            for f_idx, filename in enumerate(class_files):
                file_id = file_ids[class_idx][f_idx]
                data_id = id_mapper[file_id]
                bin_idx = math.floor((n_cv * float(permute_arr[data_id])) / len(ids))
                cv_data[bin_idx].append((filename, torch.tensor(class_idx).to(self.device)))
                self.id_sets[bin_idx].add(file_id)

        idx_sum = 0
        self.fold_idxs.append(0)
        for fold in cv_data:
            n_fold = len(fold)
            permute_arr = np.random.permutation(n_fold) + idx_sum
            self.data.update(dict(zip(permute_arr, fold)))
            idx_sum += n_fold
            self.fold_idxs.append(idx_sum)

    def split_data(self, fold=0):
        left_idx = self.fold_idxs[fold]
        right_idx = self.fold_idxs[fold+1]
        left_train = np.arange(0, left_idx)
        test_idxs = np.arange(left_idx, right_idx)
        right_train = np.arange(right_idx, self.fold_idxs[-1])
        train_idxs = np.concatenate((left_train, right_train))
        return train_idxs, test_idxs


    def get_class_sizes(self):
        return [len(class_files) for class_files in self.files_by_class]
      
    def change_mode(self, train_mode):
        self.train_mode = train_mode
        
    def get_id_sets(self):
        return self.id_sets
    
    def get_data_by_id(self, p_id):
        idxs = []
        for idx in range(self.n_data):
            filename, _ = self.data[idx]
            if p_id in filename:
                idxs.append(idx)
        
        return idxs
        
    def fill_cache(self):
        raise NotImplementedError("Please Implement fill_cache method in ParkinsonsDataset class")
    
    def normalize(self):
        raise NotImplementedError("Please Implement normalize method in ParkinsonsDataset class")

    def id_filter(self, filename):
        raise NotImplementedError("Please Implement id_filter method in ParkinsonsDataset class")

    def retrieve_file(self, filename):
        raise NotImplementedError("Please Implement retrieve_file method in ParkinsonsDataset class")

    def init_files(self):
        raise NotImplementedError("Please Implement init_files method in ParkinsonsDataset class")
        