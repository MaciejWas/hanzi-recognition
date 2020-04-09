from torch.utils.data import Dataset
from torch import from_numpy
from imbalanced import ImbalancedDatasetSampler
from label_images import RadicalOneHotEncoder, get_all_radicals
import os
from cv2 import imread
import pickle

radical_encoder = RadicalOneHotEncoder(get_all_radicals())

class RadicalsDataset(Dataset):
    """Dataset of 3-tuples: 
        * chinese character (str)
        * image of handwritten character (np.Array)
        * label (int, 1 if radical is present, 0 otherwise))
    
    Each radical requires it's own dataset, labels will be created for the radical passed in __init__.
    """

    def __init__(self, train=True, transform=None, radical=None):
        self.directory = 'train_data' if train else 'test_data'
        self.transform = transform
        self.radical = radical
        
        char_files = [(name[-1], files) for name, _, files in os.walk(self.directory)][1:] 
        mode = 'train' if train else 'test'
        dict_ = f'dict_file_{mode}.pickle' # dict_file_(train or test).pickle contains a (image filename) -> (character) dictionary.

        if dict_ not in os.listdir():
            self.file_to_char = {}
            last = len(char_files)
            print('Creating file -> char dict.')
            for i, (char, files) in enumerate(char_files):
                print(f'{i} / {last}', end='\r')
                for f in files:
                    self.file_to_char[f] = char

            with open(dict_, 'wb') as f:
                pickle.dump(self.file_to_char, f)

        else:
            with open(dict_, 'rb') as f:
                self.file_to_char = pickle.load(f)

    def __len__(self):
        n = 0
        for char in os.listdir(self.directory):
            n += len(os.listdir(os.path.join(self.directory, char)))
        return n

    def __getitem__(self, idx):
        name = str(idx) + '.png' 
        char = self.file_to_char[name]
        img = imread(os.path.join(self.directory, char, name), 0)
        label = self.radical in radical_encoder.get_rads(char)

        sample = {'char': char,
                  'img': img,
                  'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class CharacterTransform(object):
    """Normalizes image and changes it's type to torch.Tensor."""

    def __init__(self):
        pass

    def __call__(self, sample):
        img = sample['img']
        img = (img - 30) / 50
        sample['img'] = from_numpy(img).float()
        return sample

if __name__ == '__main__':
    print('Testing.')
    from torch.utils.data import DataLoader
    
    dataset = RadicalsDataset(train=False, transform=CharacterTransform(), radical='习')
    dataloader = DataLoader(
            dataset,
            batch_size=10,
            sampler=ImbalancedDatasetSampler(dataset, callback_get_label=lambda dataset, idx: dataset.__getitem__(idx)['label'])
                )

    for batch in dataloader:
        print(batch['label'], batch['char'])
