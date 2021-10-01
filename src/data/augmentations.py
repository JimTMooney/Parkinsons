import numpy as np
import torch
from scipy.signal import spectrogram


class SpecFactory():
    """
    Creates a factory function to call a spectorgram on signal data. This is used in this project as a transformation function.
    
    args:
    
    nperseg --> Length of each segment to perform an fourier transform on
    noverlap --> Number of overlapping points in each window of the spectrogram
    log --> When True, take the logarithm of each location in the resulting spectrogram
    trim --> Ratio of points to leave out in the higher frequencies.
    
    """
    def __init__(self, nperseg, noverlap, log=True, trim = 0.0):
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

class SpecMask(object):
    """
    Creates a factory function to mask blocks of a spectrogram to 0 along either the time or frequency dimension. To be used as an augmentation on spectrograms
    
    
    args:
    
    freq --> If set to True, set a block of frequencies (rows) to 0. If false, set a block of times (columns) to 0
    r --> Ratio of frequencies or times to set to 0 in a single block.
    n_blocks --> Number of masks to apply (note that each block will use r as their ratio)
    
    """
    def __init__(self, freq = True, r = .1, n_blocks=1):
        self.freq = freq
        self.r = r
        self.n_blocks = n_blocks

    def __call__(self, signal):
        n_freq = signal.shape[1]
        sz = int(n_freq * self.r)
        for _ in range(self.n_blocks):
            randint = np.random.randint(0, n_freq - sz + 1)
            if self.freq:
                signal[:, randint:randint+sz, :] = 0
            else:
                signal[:, :, randint:randint+sz] = 0

        return signal


class SpecCrop(object):
    """
    
    Factory function to be used as a transform for spectrograms. This class randomly crops out the same sized block from each spectrogram along the time dimension. If the time dimension is less than this standard size, then zeros are appended along the time dimension.
    
    args:
    
    sz --> The length of the time dimension (i.e. the number of fourier transforms considered).
    
    """
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