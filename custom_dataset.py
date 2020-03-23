from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torch import from_numpy
from imbalanced import ImbalancedDatasetSampler
from torch.nn.functional import interpolate
from label_images import *
import os
from skimage import io 
import pickle

e = RadicalOneHotEncoder(get_all_radicals())

class RadicalsDataset(Dataset):
    """Character to radicals dataset."""

    def __init__(self, train=True, transform=None, radical=None):
        self.directory = 'train_data' if train else 'test_data'
        self.transform = transform
        self.radical = radical
        char_files = [(name[-1], files) for name, _, files in os.walk(self.directory)][1:]
        
        self.file_to_char_label = {}
        
        last = len(char_files)
        print('Creating file -> char dict')
        for i, (char, files) in enumerate(char_files):
            print(f'{i} / {last}', end='\r')
            for f in files:
                label = radical in e.partial_decode(e.encode(char))
                self.file_to_char_label[f] = (char, int(label))
        
        print('Done initializing class.')


    def __len__(self):
        n = 0
        for char in os.listdir(self.directory):
            n += len(
                    os.listdir(os.path.join(self.directory, char))
                    )
        return n

    def __getitem__(self, idx):
        name = str(idx) + '.png' 
        char, label = self.file_to_char_label[name]
        img = io.imread(os.path.join(self.directory, char, name))
        
        sample = {'char': char,
                  'img': img,
                  'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class CharacterTransform(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        img = (img - 30) / 50
       
        sample['img'] = from_numpy(img).float()
        #sample['label'] = from_numpy(label).float()
        
        return sample

if __name__ == '__main__':
    print('Testing.')
    
    dataset = RadicalsDataset(train=False, transform=CharacterTransform(), radical='习')
    dataloader = DataLoader(dataset, batch_size=10, sampler=ImbalancedDatasetSampler(dataset, callback_get_label=lambda dataset, idx: dataset.__getitem__(idx)['label']))

    for batch in dataloader:
        print(batch['label'], batch['char'])
