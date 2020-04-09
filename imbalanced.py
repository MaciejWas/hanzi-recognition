import torch
import torch.utils.data
import torchvision
import os
from label_images import RadicalOneHotEncoder, get_all_radicals
import pickle
import numpy as np

radical_encoder = RadicalOneHotEncoder(get_all_radicals())

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):
        radical = dataset.radical
        radical_n = int(
                np.argwhere(radical_encoder.rads == radical)
                )
        mode = 'train' if dataset.directory == 'train_data' else 'test'
        self.indices = list(range(len(dataset))) 
        self.callback_get_label = callback_get_label
        self.num_samples = len(dataset) 
         
        labels = np.load(os.path.join('labels', f'{mode}_labels.npz'), allow_pickle=True)['arr_0'][()][:, radical_n]
        ones_count = np.sum(labels)
        label_to_count = {
                1: ones_count,
                0: len(dataset) - ones
                }
        
        weights = np.ones(len(self.indices))
        weights[np.argwhere(labels == 1)] = 1/label_to_count[1]
        weights[np.argwhere(labels != 1)] = 1/label_to_count[0]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return self.callback_get_label(dataset, idx)
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
