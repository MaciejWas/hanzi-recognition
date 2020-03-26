import numpy as np
import cv2
import os, sys
import struct
from codecs import decode
import numpy as np
import pickle

def load_gnt_file(filename):
    """
    Loads characters and images from a given GNT file.
    params:
        filename: The file path to load.
    returns:
        (image: Pillow.Image.Image, character) tuples
    """

    with open(filename, "rb") as f:
        while True:
            packed_length = f.read(4)
            if packed_length == b'':
                break
            length = struct.unpack("<I", packed_length)[0]
            raw_label = struct.unpack(">cc", f.read(2))
            width = struct.unpack("<H", f.read(2))[0]
            height = struct.unpack("<H", f.read(2))[0]
            photo_bytes = struct.unpack("{}B".format(height * width), f.read(height * width))
            label = decode(raw_label[0] + raw_label[1], encoding="gb2312")
            yield np.array(photo_bytes).reshape(height, width), label

def load_chunk(filename):
    """There's ~3500 compressed images in a single gnt file.
    This function produces a list of (img, label) tuples (one for every image
    in the .gnt file)."""
    return list(load_gnt_file(filename))

def create_category_folders(target):
    targetdir = unloaded_datasets[target]

    if 'chars.pickle' in os.listdir():
        with open('chars.pickle', 'rb') as handle:
            characters_info = pickle.load(handle)
    else:
        print('File "chars.pickle" not found. Run get_statistics.py')
        sys.exit(1)

    unique_characters = list(characters_info.keys())

    os.makedirs(targetdir, exist_ok=True)

    for char in unique_characters:
        os.makedirs(
            os.path.join(targetdir, char),
            exist_ok=True
            )
    return targetdir

compressed_datasets = {'train': ['HWDB1.1trn_gnt_P1', 'HWDB1.1trn_gnt_P2', 'competition-gnt'],
            'test' : ['HWDB1.1tst_gnt']}

unloaded_datasets = {'train': 'train_data',
            'test': 'test_data'}

info = """Usage:
    $ python3 create_images.py test
    or
    $ python3 create_images.py train"""


if __name__ == '__main__':
    try:
        target = sys.argv[1]
    except (KeyError, IndexError):
        print(info)
        sys.exit(2)

    targetdir = create_category_folders(target)
    print(f'Creating {target} data in {targetdir} directory. It\'s going to take a while.')

    n = 0
    for dir_ in compressed_datasets[target]:
        all_gnt_files = os.listdir(
                os.path.join('data', dir_)
                )
        for gnt_file in all_gnt_files:
            print('Converting',f'data/{dir_}/{gnt_file}', end='\r')
            path_from = os.path.join('data', dir_, gnt_file)
            chunk = load_chunk(path_from)
            for img, char in chunk:
                cv2.imwrite(
                    os.path.join(targetdir, char, str(n) + '.png'), img
                    )
                n+=1

    print('Done')
    sys.exit(0)
