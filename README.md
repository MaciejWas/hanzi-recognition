## Chinese character radical detection
A chinese character is composed of smaller ones called 'radicals'. My aim was to create a NN for detection of radicals (one model, but with different weights for different radicals) in handwritten chinese characters. For this task I used a simplified version of Alexnet and the CASIA dataset of handwritten characters.

## Dataset
http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html

~3500 characters, ~300 examples of each (in the training data)

#### Examples
![alt text](readme_images/collage.png "character collage")

## Results
Below is a chart of accuracy and recall on training data over number of batches. Size of training batch in this case was 512, and the model was training for recognizing radical 耳. Model archieves similar similar values of accuracy and recall for most of radical components, some of them are available in data_out directory.
#### Accuracy
![alt text](readme_images/accuracy.svg "accuracy score")
#### Recall
![alt text](readme_images/recall.svg "recall score")
#### Test Accuracy
![alt text](readme_images/test_accuracy.svg "recall score")

## Licensed modules
I am using a modified version of **PyCasia** python module:
https://github.com/lucaskjaero/PyCasia
licensed under Apache 2.0 License,

as well as **Imbalanced Data Sampler**:
https://github.com/ufoym/imbalanced-dataset-sampler
licensed under MIT License.

## How to run
To start, you don't have to manually download anything outside this repository, except for python libraries listed in *requirements.txt*.
#### Run in this order:
##### get_dataset.py -> unpack_images.py -> label_images.py -> custom_dataset.py
After you do, every file needed should be in the hanzi-recognition directory. After unpacking and deleting compressed datasets, size of the dir should be ~**11GB**.

