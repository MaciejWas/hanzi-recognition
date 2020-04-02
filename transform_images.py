from PIL import Image, ImageOps
import numpy as np
import os, sys

width, height = 110, 110

def transform(image_pil, width, height):
    '''Resizes and inverts PIL image while keeping image's ratio.
    '''
    ratio_w = width / image_pil.width
    ratio_h = height / image_pil.height
    if ratio_w < ratio_h:
        # It must be fixed by width
        resize_width = width
        resize_height = round(ratio_w * image_pil.height)
    else:
        # Fixed by height
        resize_width = round(ratio_h * image_pil.width)
        resize_height = height

    image_resize = image_pil.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('L', (width, height), 255)
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset)
    return ImageOps.invert(background.convert('L'))

info = """Usage:
$ python3 create_labele.py test
or
$ python3 create_labels.py train"""

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(info)
        sys.exit(1)
    elif sys.argv[1] not in ['train', 'test']:
        print(info)
        sys.exit(1)

    directory = {'train': 'train_data', 'test': 'test_data'}[sys.argv[1]]

    chars = os.listdir(directory)
    for char in chars:
        for f in os.listdir(os.path.join(directory, char)):
            filepath = os.path.join(directory, char, f)
            print(f'transforming {filepath}', end='\r')
            img = Image.open(filepath)

            if np.array(img).shape != (110, 110):
                img_transformed = transform(img, width, height)
                img_transformed.save(filepath)  
            elif np.array(img).mean() < 165: # Checks if image is inverted
                continue
            else:
                img_transformed = transform(img, width, height)
                img_transformed.save(filepath)
