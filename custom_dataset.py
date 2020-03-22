from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torch import from_numpy
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
        self.radical_id = np.argwhere(e.rads == self.radical)
        assert bool(self.radical_id)
        char_files = [(name[-1], files) for name, _, files in os.walk(self.directory)][1:]
        self.file_to_char_label = {}
        
        last = len(char_files)
        print('Creating file -> char dict')
        for i, (char, files) in enumerate(char_files):
            print(f'{i} / {last}', end='\r')
            for f in files:
                if radical: 
                    label = e.encode(char)[self.radical_id].reshape(1)
                    self.file_to_char_label[f] = (char, label)
                else:
                    self.file_to_char_label[f] = (char, e.encode(char))
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
        label = label.astype(np.int8)
        sample = {'img': img,
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
        img = from_numpy(img).float()
        
        sample['label'] = from_numpy(label).float()
        sample['img'] = interpolate(img.reshape(1,1,110,110), scale_factor=2.07, mode='bilinear', align_corners=True).reshape(1, 227, 227)
        
        return sample

if __name__ == '__main__':
    from torch import nn
    print('Testing.')
    
    if 'testing.pickle' not in os.listdir():
        dataset = RadicalsDataset(train=False, transform=CharacterTransform(), radical='止')
        print('Created dataset instance.')
        with open('testing.pickle', 'wb') as f:
            pickle.dump(dataset, f)
    else:        
        with open('testing.pickle', 'rb') as f:
            dataset = pickle.load(f)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in dataloader:
        print(batch['label'])
        if 1 in batch['label']:
            break
