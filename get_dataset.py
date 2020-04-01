# Run this script first.
# Will download compressed dataset in ./data directory, as well as create chars.pickle and sizes.pickle.

import numpy as np
from pycasia.CASIA import CASIA
import sys, os
from itertools import chain
import pickle

dataset_manager = CASIA(path='data')

# Creating a generator yielding pairs of (image, character)
training_generators = [
    dataset_manager.load_dataset('HWDB1.1trn_gnt_P1'),
    dataset_manager.load_dataset('HWDB1.1trn_gnt_P2'),
    dataset_manager.load_dataset('competition-gnt')
]
 
testing_pairs_gen = dataset_manager.load_dataset('HWDB1.1tst_gnt')
training_pairs_generator = chain(*[gen for gen in training_generators])

sizes = {} # keys are tuples of (width, height), values are numbers of images
           # with those dimensions
characters = {} # keys are chinese characters, values are numbers of their occurences

if __name__ == '__main__':
    generator = chain(training_pairs_generator, testing_pairs_gen)
    
    if 'sizes.pickle' in os.listdir():
        input('sizes.pickle already created. Continue? CTRL + C to abort')

    if 'chars.pickle' in os.listdir():
        input('chars.pickle already created. Continue? CTRL + C to abort')

    for (i, (img, char)) in enumerate(generator):
        print(i, char, end='\r')
        if img.shape not in sizes.keys():
            sizes[img.shape] = 1
        else:
            sizes[img.shape] += 1

        if char not in characters.keys():
            characters[char] = 1
        else:
            characters[char] += 1

    print(f'Total characters processed: {i}')

    with open('sizes.pickle', 'wb') as handle:
        pickle.dump(sizes, handle)

    with open('chars.pickle', 'wb') as handle:
        pickle.dump(characters, handle)

