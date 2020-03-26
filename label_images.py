# In order to succesfully create labels for examples, you must have
# previously ran get_dataset.py and unpack_images.py (in this order).
# the data foler should look like this:
# /training_data
#      | - training_data/一/ (examples)
#      |
#      | - training_data/丁/ (examples)
#
#                   ...
#
#      |
#      |- training_data/龟/ (examples)
#
# The same with train_data folder


import os
import pickle
from zhon import hanzi, cedict
import re
import numpy as np
import sys
from scipy import sparse

redundant = ['⿰','⿱','⿲','⿳','⿴','⿵','⿶','⿷','⿸','⿹','⿺','⿻',]

def create_simplified_IDS():
    if 'IDS_dictionary_simp.txt' in os.listdir():
        return None

    with open('IDS_dictionary.txt', 'r') as f:
        lines = f.read().split('\n')[:-1]

    chars = os.listdir('train_data')

    with open('IDS_dictionary_simp.txt', 'w') as f:
        f.writelines(
        [line + '\n' for line in lines if process_line(line)[0] in chars]
        )

    print('Created simplified IDS file')

def find_example_n(directory):
    n = 0
    for subdir in os.listdir(directory):
        files_in_dir = len(
            os.listdir(os.path.join(directory, subdir))
            )
        n += files_in_dir
    return n

def process_line(line, radicals_only=True):
    """Processes a line from IDS_dictionary.
    INPUT: a line from IDS_dictionary
    OUTPUT: * if process_rest is True returns a pair of character and
                list of the character's radicals (with repetitions!)):
                '児:⿱ ⿰ 丨 日 儿\n'   ->   ('児', ['丨', '日', '儿'])
            * if process_rest is False returns a pair of character and it's
                radical decomposition eg.
                '児:⿱ ⿰ 丨 日 儿\n'   ->   ('児', '⿱ ⿰ 丨 日 儿')
    """
    char, rest = line.split(':')

    decomp = rest.split(' ')
    processed = []
    for x in decomp:
        if x == '':
            continue
        elif radicals_only == True and x in redundant:
            continue
        else:
            processed.append(x)

    return (char, processed)

def get_radical_dict(radicals_only=True):
    """Applies process_line() on all lines in the IDS_dictionary_simp.txt.
        Returns:
            dict: character -> radicals     
            OR (depending on second argument in process_line())
            dict: character -> character's radical composition
            """
    with open('IDS_dictionary_simp.txt', 'r') as f:
        lines = f.read().split('\n')[:-1]

    ids_dict = dict(zip(
        [process_line(line, radicals_only=radicals_only)[0] for line in lines],
        [process_line(line, radicals_only=radicals_only)[1] for line in lines]
        ))

    return ids_dict

def get_all_radicals():
    ids_dict = get_radical_dict()
    all_radicals = []
    chars = os.listdir('train_data')

    for char in chars:
        for radical in ids_dict[char]:
            if radical not in all_radicals:
                all_radicals.append(radical)

    return np.array(all_radicals, dtype=object)

class RadicalOneHotEncoder:
    def __init__(self, all_radicals):
        self.rads = all_radicals
        self.radicals_n = len(all_radicals)
        self.char_to_rads = get_radical_dict()

    def encode(self, character):
        y = np.zeros(self.radicals_n, dtype=np.bool_)

        for radical in self.char_to_rads[character]:
            pos = np.argwhere(self.rads == radical)
            y[pos] = True
        return y

    def partial_decode(self, label):
        mask = label == 1
        return self.rads[mask]

info = """Usage:
$ python3 create_labels.py test
or
$ python3 create_labels.py train"""


if __name__ == '__main__':
    try:
        directory = {'train': 'train_data', 'test': 'test_data'}[sys.argv[1]]
    except (IndexError, KeyError):
        print(info)

    create_simplified_IDS()
    n_examples = find_example_n(directory)
    all_radicals = get_all_radicals()
    encoder = RadicalOneHotEncoder(all_radicals)

    labels = sparse.lil_matrix(
        np.zeros(shape=(n_examples, len(all_radicals)), dtype=np.int8)
    )

    if f'{sys.argv[1]}_labels' in os.listdir('labels'):
        input('Labels already there. Continue? CTRL+C to abort.')

    for dir in os.listdir(directory):
        examples_in_dir = os.listdir(os.path.join(directory, dir))
        example_ids = [example[:-4] for example in examples_in_dir]
        for n in example_ids:
            print(dir, n, end='\r')
            n = int(n)
            labels[n, :] = encoder.encode(dir)
        
    if 'labels' not in os.listdir():
        os.makedirs('labels')

    np.savez(f'labels/{sys.argv[1]}_labels', labels)
