import os, sys
import pickle
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from tensorboardX import SummaryWriter
from custom_dataset import RadicalsDataset, CharacterTransform, e, ImbalancedDatasetSampler
#from accuracy_metric import average_misclassified_radicals
import numpy as np
from models import Model, balanced_loss 


# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('cuda:', torch.cuda.is_available())


assert sys.argv[1] in e.rads

WARM = True
RADICAL = sys.argv[1]
NUM_EPOCHS = 3  # original paper
BATCH_SIZE = 512
MOMENTUM = 0.9
LR_DECAY = 0.0005
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

    tbwriter = SummaryWriter(log_dir=LOG_DIR)
    print('TensorboardX summary writer created')

    # create model
    model = Model()
    
    if WARM:
        try:
            model.load_state_dict(torch.load('alexnet_data_out/火/models/model火_at_state16.pkl'), strict=False)
            print('Warm start with 火-model.')
        except:
            print('Unable to load previous model for warm start. Continuing with random weights.')

    model.to(device)

    print('Neural network created')
    
    # ------ Loading dataset for radical detection ------
    dataset = RadicalsDataset(
        train=True,
        transform=CharacterTransform(),
        radical=RADICAL,
            )

    test_dataset = RadicalsDataset(
        train=False,
        transform=CharacterTransform(),
        radical=RADICAL,
            )
    print('Dataset created')

    dataloader = data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=6,
        sampler=ImbalancedDatasetSampler(dataset, callback_get_label=lambda dataset, idx: dataset.__getitem__(idx)['label'])
            ) 
    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=6,
        shuffle=True
            )

    print('Dataloader created')
    
    optimizer = optim.Adam(params=model.parameters(), lr=0.00002)
    
    print('Optimizer created')
    
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    criterion = nn.BCEWithLogitsLoss()
    print('LR Scheduler created')

    print('Starting training...')

    max_steps = 2200
    total_steps = 1
    for epoch in range(NUM_EPOCHS):
        for batch in dataloader:
            imgs = batch['img']
            classes = batch['label']
            imgs, classes = imgs.to(device), classes.to(device)
            output = model(imgs)

            classes = classes.type_as(output)
            
            loss = criterion(output, classes)
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            
            if total_steps % 10 == 0:
                with torch.no_grad():
                    preds = output > 0
                    accuracy = torch.sum(preds == classes).item() / BATCH_SIZE
                    recall = torch.sum((preds == classes) * (classes == 1)).item() / torch.sum(classes == 1).item()

                    print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}, \tRec: {}'
                        .format(epoch + 1, total_steps, loss.item(), accuracy, recall))
                    
                    tbwriter.add_scalar('loss', loss.item(), total_steps)
                    tbwriter.add_scalar('accuracy', accuracy, total_steps)
                    tbwriter.add_scalar('recall', recall, total_steps)

            if total_steps % 100 == 0:
                with torch.no_grad():
                    print('*' * 10)
                    for name, parameter in model.named_parameters():
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
            if total_steps % 500 == 0 or total_steps == 1:
                with torch.no_grad():
                    print('Evaluating on ten random batches from testing data.')
                    for k, batch in enumerate(test_dataloader):
                        imgs = batch['img']
                        classes = batch['label']

                        imgs, classes = imgs.to(device), classes.to(device)
                        output = model(imgs)

                        classes = classes.type_as(output)

                        preds = output > 0
                        accuracy = torch.sum(preds == classes).item() / BATCH_SIZE
                        recall = torch.sum((preds == classes) * (classes == 1)).item() / torch.sum(classes == 1).item()
                        print('test accuracy:', accuracy, '\t recall:', recall)
            
                        tbwriter.add_scalar('test_loss', loss.item(), total_steps)
                        tbwriter.add_scalar('test_accuracy', accuracy, total_steps)
                        tbwriter.add_scalar('test_recall', recall, total_steps)
                        
                        if k == 9:
                            break
            total_steps += 1
        lr_scheduler.step()

    # save weak model
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model{RADICAL}_at_state{total_steps}.pkl')
    state = {
        'epoch': epoch,
        'total_steps': total_steps,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict()
    }
    torch.save(state, checkpoint_path)
    sys.exit(0)
