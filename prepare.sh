#!/bin/bash
python3 get_statistics.py 
python3 create_images.py train
python3 create_images.py test
python3 create_labels.py train
python3 create_labels.py test
python3 transform_images.py train
python3 transform_images.py test
