import numpy as np
import torch
from scipy.signal import spectrogram


class SpecFactory():
    def __init__(self, nperseg, noverlap, log=True, trim =0.0):
        self.nperseg = nperseg 
        self.noverlap = noverlap
        self.log = log
        self.trim = trim
        self.trim_idx = None
        

    def __call__(self, signal):
        _, _, Sxx = spectrogram(signal.T, fs=1e3, nperseg=self.nperseg, noverlap=self.noverlap)
        if self.trim_idx == None:
            self.trim_idx = Sxx.shape[1] - int(Sxx.shape[1] * self.trim)
        Sxx = Sxx[:, self.trim_idx:, :]
        if self.log:
            epsilon = np.finfo(float).eps
            return np.log(Sxx + epsilon)
        else:
            return Sxx

class FreqMask(object):
    def __init__(self, r = .1, n_blocks=1):
        self.r = r
        self.n_blocks = n_blocks

    def __call__(self, signal):
        n_freq = signal.shape[1]
        sz = int(n_freq * self.r)
        for _ in range(self.n_blocks):
            randint = np.random.randint(0, n_freq - sz + 1)
            signal[:, randint:randint+sz, :] = 0

        return signal

class TimeMask(object):
    def __init__(self, r=.1, n_blocks=1):
        self.r = r
        self.n_blocks = n_blocks

    def __call__(self, signal):
        n_time = signal.shape[2]
        sz = int(n_time * self.r)
        for _ in range(self.n_blocks):
            randint = np.random.randint(0, n_time - sz + 1)
            signal[:, :, randint:randint+sz] = 0

        return signal

class SpecCrop(object):
    def __init__(self, sz=10):
        self.sz = sz
        
    def __call__(self, spec):
        spec_sz = spec.shape
        diff = spec_sz[2] - self.sz
        spec_seg = spec
        if (diff < 0):
            zeros = torch.zeros((spec_sz[0], spec_sz[1], -diff))
            if spec.is_cuda:
                device = torch.get_device(spec)
                zeros = zeros.to(device)
            spec_seg = torch.cat((spec, zeros), 2)
        elif (diff > 0):
            randint = np.random.randint(0, diff+1)
            spec_seg = spec[:, :, randint:randint+self.sz]
            
        return spec_seg