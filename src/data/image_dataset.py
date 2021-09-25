import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from parkinsons_dataset import ParkinsonsDataset
from torchvision import transforms
import numpy as np
from PIL import Image


class ImageDataset(ParkinsonsDataset):
    def __init__(self, root_dir, test_type, device, train_transform=None, test_transform=None, 
                 preprocess=None, cache=False):
        super().__init__()

        self.device = device
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.train_transform = train_transform 
        self.test_transform = test_transform 
        
        self.cache = cache
        self.cached_data = {}


        self.init_files(test_type)
        self.cv_init()
        self.normalize()
        self.fill_cache()
        
    def __getitem__(self, idx):
        filename, label = self.data[idx]

        data = None
        if self.cache:
            data = self.cached_data[idx]
        else:
            data = self.retrieve_file(filename)
            data = data.to(self.device)
            if self.preprocess is not None:
                data = self.preprocess(data)
            data = self.normal_transform(data)
        
        if self.train_mode and self.train_transform is not None:
            data = self.train_transform(data)
        elif self.test_transform is not None:
            data = self.test_transform(data)

        return data, label
        
    
    def fill_cache(self):
        if self.cache:
            for idx in range(self.n_data):
                filename, label = self.data[idx]
                data = self.retrieve_file(filename)
                if self.preprocess is not None:
                    data = self.preprocess(data)
                data = self.normal_transform(data)
                self.cached_data[idx] = data.to(self.device)
        
    def normalize(self):
        normal_idxs=(1, 2)
        mean = 0
        std = 0
        
        for idx in range(self.n_data):
            filename, label = self.data[idx]
            data = self.retrieve_file(filename)
            if self.preprocess is not None:
                data = self.preprocess(data)
            
            mean += data.mean(normal_idxs)
            std += data.std(normal_idxs)
        
        mean /= self.n_data
        std /= self.n_data
        
        self.normal_transform = transforms.Normalize(mean, std)
        
    def id_filter(self, filename):
        tail = os.path.split(filename)[1]
        idx = tail.split('-')[1].split('.')[0]
        return idx + '.'

    def retrieve_file(self, filename):
        img = Image.open(filename)
        img = np.asarray(img).astype(float)
        return (torch.from_numpy(img)).permute((2, 0, 1)).float()

    def init_files(self, test_type):
        controls = self.get_file_list('Healthy' + test_type)
        patients = self.get_file_list('Patient' + test_type)


        self.files_by_class.append(controls)
        self.files_by_class.append(patients)

        self.n_data = len(controls) + len(patients)

    def get_file_list(self, f_string):
        file_loc = os.path.join(self.root_dir, f_string)
        return [os.path.join(file_loc, f) for f in os.listdir(file_loc) if f.endswith('.jpg')]