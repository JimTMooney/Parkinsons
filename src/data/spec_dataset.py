import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)


from parkinsons_dataset import ParkinsonsDataset
import torch
from torchvision import transforms



class SpecDataset(ParkinsonsDataset):
    """
    
    pytorch Dataset for holding spirals or meanders for training.
    
    
    args:
    
    root_dir --> Location of the images directory. Subdirectories of this directory should be 'HealthyMeander', 'HealthySpiral', 'PatientMeander', 'PatientSpiral'
    test_type --> If 'Meander', then use meander signals for this dataset, if 'Spiral', then use spiral signals.
    device --> Device where training will occur.
    sig_transform --> Augmentations  to use on the raw signal
    spec --> SpecFactory function to create spectrograms.
    train_transform --> The transforms used during training.
    test_transform --> The transforms used during testing and validation.
    cache --> If True, put everything on device. If False, load separately each time.
    
    """
    def __init__(self, root_dir, test_type, device, sig_transform = None, spec = None, 
                 train_transform = None, test_transform = None, cache = False):
        super().__init__()
        
        self.root_dir = root_dir
        self.test_type = test_type
        self.device = device
        self.sig_transform = sig_transform
        self.spec = spec
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.cache = cache
        
        self.cached_data = {}
        
        self.init_files()
        self.cv_init()
        self.normalize()
        self.fill_cache()
        
        
    def __getitem__(self, idx):
        filename, label = self.data[idx]
        
        data = None
        if self.cache:
            data = self.cached_data[idx]
            if self.sig_transform is not None:
                data = self.sig_transform(data)
                data = self.spec(data)
                data = data.to(self.device, dtype=torch.float)
                data = self.normal_transform(data)
        else:
            data = self.retreive_file(filename)
            if self.sig_transform is not None:
                data = self.sig_transform(data)
            data = self.spec(data)
            data = data.to(self.device, dtype=torch.float)
            data = self.normal_transform(data)
        
        if self.train_mode and self.train_transform is not None:
            data = self.train_transform(data)
        elif self.test_transform is not None:
            data = self.test_transform(data)

        return data.float(), label
    
    def fill_cache(self):
        if self.cache:
            for idx in range(self.n_data):
                filename, label = self.data[idx]
                data = self.retreive_file(filename)
                if self.sig_transform is None:
                    data = self.spec(data)
                    data = self.normal_transform(torch.tensor(data))
                data = data.to(self.device, dtype=torch.float)
                self.cached_data[idx] = data.to(self.device)
        
    def normalize(self):
        normal_idxs=(1, 2)
        mean = 0
        std = 0
        
        for idx in range(self.n_data):
            filename, label = self.data[idx]
            data = self.retreive_file(filename)
            data = self.spec(data)
            
            mean += data.mean(normal_idxs)
            std += data.std(normal_idxs)
        
        mean /= self.n_data
        std /= self.n_data
        
        self.normal_transform = transforms.Normalize(mean, std)
        
    
    def id_filter(self, filename):
        tail = os.path.split(filename)[1]
        idx = tail.split('-')[1].split('.')[0]
        return idx + '.'
    
    def retreive_file(self, filename):
        f = open(filename)

        f_signals = []
        for idx, line in enumerate(f):
            if '#' not in line: 
                line = line.split('\t')
                l = [float(sig) for sig in line]
                f_signals.append(l)

        return torch.tensor(f_signals)
                
        
    def init_files(self):
        controls = self.get_file_list('Healthy')
        patients = self.get_file_list('Patients')


        self.files_by_class.append(controls)
        self.files_by_class.append(patients)

        self.n_data = len(controls) + len(patients)
                
    def get_file_list(self, f_string):
        file_loc = os.path.join(self.root_dir, f_string, 'Signal')
        return [os.path.join(file_loc, f) for f in os.listdir(file_loc) if f.endswith('.txt') and self.test_type in f]