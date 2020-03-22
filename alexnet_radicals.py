import os, sys
import pickle
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from tensorboardX import SummaryWriter
from custom_dataset import RadicalsDataset, CharacterTransform, e
#from accuracy_metric import average_misclassified_radicals
import numpy as np
from alexnet import AlexNet, balanced_loss 


# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('cuda:', torch.cuda.is_available())

try:
    radical = sys.argv[1]
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
BATCH_SIZE = 512
MOMENTUM = 0.5
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227  # pixels
DEVICE_IDS = [0]  # GPUs to use
# modify this to point to your data directory
INPUT_ROOT_DIR = None
OUTPUT_DIR = os.path.join('alexnet_data_out', RADICAL)
LOG_DIR = os.path.join(OUTPUT_DIR, 'tblogs')  # tensorboard logs
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'models')  # model checkpoints

# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        batch_size=BATCH_SIZE,
        num_workers=6
            )
    test_dataloader = data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=6
            )

    print('Dataloader created')
    
    optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
    
    print('Optimizer created')

    # multiply LR by 1 / 10 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss(
            torch.tensor([1, 20]).to(device)
            )
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.MultiLabelMarginLoss()
    print('LR Scheduler created')

    print('Starting training...')
    total_steps = 1
    for epoch in range(NUM_EPOCHS):
        for batch in dataloader:
            imgs = batch['img']
            classes = batch['label']

            imgs, classes = imgs.to(device), classes.to(device)
            output = alexnet(imgs)
            #print(output.cpu()) 
            #loss = F.binary_cross_entropy(output, classes)

            loss = balanced_loss(output, classes)
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            
            if total_steps % 10 == 0:
                with torch.no_grad():
                    preds = output > .5
                    accuracy = torch.sum(preds == classes)
                    print('accuracy:', accuracy)
                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                        .format(epoch + 1, total_steps, loss.item(), accuracy.item()))
                    tbwriter.add_scalar('loss', loss.item(), total_steps)
                    tbwriter.add_scalar('accuracy', accuracy.item(), total_steps)

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
