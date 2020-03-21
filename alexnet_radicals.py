"""
Implementation of AlexNet, from paper
"ImageNet Classification with Deep Convolutional Neural Networks" by Alex Krizhevsky et al.
See: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
"""
import os, sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from tensorboardX import SummaryWriter
from custom_dataset import RadicalsDataset, CharacterTransform, e
#from accuracy_metric import average_misclassified_radicals
import numpy as np

# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('cuda:', torch.cuda.is_available())

# define model parameters

try:
    radical = sys.argv[1]
    print(radical)
    print(e.rads)
    if sys.argv[1] in e.rads:
        print(' --- Training network for detecring radical', radical, '---')
    else:
        print(f'Usage: $ python3 {__file__}.py [radical character]')
        sys.exit(1)
except IndexError:
    print(f'Usage: $ python3 {__file__}.py [radical character]')
    sys.exit(1)

RADICAL = sys.argv[1]
NUM_EPOCHS = 90  # original paper
BATCH_SIZE = 2
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227  # pixels
DEVICE_IDS = [0]  # GPUs to use
# modify this to point to your data directory
INPUT_ROOT_DIR = None
OUTPUT_DIR = 'alexnet_data_out'
LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints

# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """
    def __init__(self):
        """
        Define and allocate layers for this neural net.
        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        """
        Define and allocate layers for this neural net.
        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
            )

        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.35, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1),
        )
        self.init_bias()  # initialize bias

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        x = self.net(x)
        #print(x.shape)
        x = x.view(-1, 256 * 6 * 6)  # reduce the dimensions for linear layer input
        x = self.classifier(x)
        #test = x.cpu().detach().numpy()
        return x

if __name__ == '__main__':
    # print the seed value
    seed = torch.initial_seed()
    print('Used seed : {}'.format(seed))

    tbwriter = SummaryWriter(log_dir=LOG_DIR)
    print('TensorboardX summary writer created')

    # create model
    alexnet = AlexNet().to(device)
    # train on multiple GPUs
    alexnet = torch.nn.parallel.DataParallel(alexnet, device_ids=DEVICE_IDS)
    print(alexnet)
    print('AlexNet created')
    
    # ------ Loading dataset for radical detection ------
    if 'dataset.pickle' not in os.listdir() or 'test_dataset.pickle' not in os.listdir():
        dataset = RadicalsDataset(
            train=True,
            transform=CharacterTransform(),
            radical=RADICAL
                )

        test_dataset = RadicalsDataset(
                train=False,
                transform=CharacterTransform(),
                radical=RADICAL
                )
        with open('dataset.pickle', 'wb') as f:
            pickle.dump(dataset, f)
        with open('test_dataset.pickle', 'wb') as f:
            pickle.dump(test_dataset, f)
    else:
        with open('dataset.pickle', 'rb') as f:
            dataset = pickle.load(f)
        with open('test_dataset.pickle', 'rb') as f:
            test_dataset = pickle.load(f)
    
    print('Dataset created')

    dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=BATCH_SIZE
            )
    test_dataloader = data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE
            )

    print('Dataloader created')
    
    optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
    
    print('Optimizer created')

    # multiply LR by 1 / 10 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.MultiLabelMarginLoss()
    print('LR Scheduler created')

    print('Starting training...')
    total_steps = 1
    for epoch in range(NUM_EPOCHS):
        for batch in dataloader:
            imgs = batch['img']
            classes = batch['label']

            print('loaded:', imgs.shape, classes.shape)

            imgs, classes = imgs.to(device), classes.to(device)
            output = alexnet(imgs)
            
            print('after alexnet:', output.shape)
            loss = criterion(output, classes)
    
            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
   
            # log the information and add to tensorboard
            if total_steps % 10 == 0:
                with torch.no_grad():
                    preds = output > .5
                    accuracy = torch.sum(preds == classes)

                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                        .format(epoch + 1, total_steps, loss.item(), accuracy))
                    tbwriter.add_scalar('loss', loss.item(), total_steps)
                    tbwriter.add_scalar('accuracy', accuracy, total_steps)

            # print out gradient values and parameter average values
            if total_steps % 100 == 0:
                with torch.no_grad():
                    # print and save the grad of the parameters
                    # also print and save parameter values
                    print('*' * 10)
                    for name, parameter in alexnet.named_parameters():
                        if parameter.grad is not None:
                            avg_grad = torch.mean(parameter.grad)
                            print('\t{} - grad_avg: {}'.format(name, avg_grad))
                            tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
                            tbwriter.add_histogram('grad/{}'.format(name),
                                    parameter.grad.cpu().numpy(), total_steps)
                        if parameter.data is not None:
                            avg_weight = torch.mean(parameter.data)
                            print('\t{} - param_avg: {}'.format(name, avg_weight))
                            tbwriter.add_histogram('weight/{}'.format(name),
                                    parameter.data.cpu().numpy(), total_steps)
                            tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)
            total_steps += 1

        lr_scheduler.step()

        # save checkpoints
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'alexnet_states_e{}.pkl'.format(epoch + 1))
        state = {
            'epoch': epoch,
            'total_steps': total_steps,
            'optimizer': optimizer.state_dict(),
            'model': alexnet.state_dict(),
            'seed': seed,
        }
        torch.save(state, checkpoint_path)
