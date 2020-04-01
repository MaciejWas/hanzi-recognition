import os, sys
import pickle
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from tensorboardX import SummaryWriter
from custom_dataset import RadicalsDataset, CharacterTransform, e, ImbalancedDatasetSampler
import numpy as np
from models import Model, balanced_loss 


def evaluate_on_test_set(model, dataloader):
    """Batches with even indices belong to validation set, odd - to test set.
    Because shuffle parameter in dataloader is set to False, validation and teest set will be the same for all radicals."""
    n_corr_classified = 0
    n_images = 0

    n_corr_classified_and_class_instance = 0
    n_class_instances = 0

    for k, batch in enumerate(dataloader):

        if k % 2 != 0: # I treat batches with even indices as test set
            continue

        imgs = batch['img']
        classes = batch['label']

        imgs, classes = imgs.to(device), classes.to(device)
        output = model(imgs)
        classes = classes.type_as(output)

        preds = output > 0

        n_corr_classified += torch.sum(preds == classes).item()
        n_images += len(batch)

        n_corr_classified_and_class_instance += torch.sum((preds == classes) * (classes == 1)).item()
        n_class_instances += torch.sum(classes == 1).item()
    try:
        recall = n_corr_classified_and_class_instance / n_class_instances
        accuracy = n_corr_classified / n_images

    except ZeroDivisionError:
        print(f'No examples of {radical} in testing data!')
        recall = 1
    
    return accuracy, recall

def train_model(radical, warm, num_epochs, batch_size, lr_decay):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('cuda:', torch.cuda.is_available())

    OUTPUT_DIR = os.path.join('data_out', radical)
    LOG_DIR = os.path.join(OUTPUT_DIR, 'tblogs')  
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'models')  

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True) 

    tbwriter = SummaryWriter(log_dir=LOG_DIR)
    print('TensorboardX summary writer created')

    model = Model()
    
    if warm:
        try:
            model.load_state_dict(torch.load('data_out/火/models/model火_at_state16.pkl'), strict=False)
            print('Warm start with 火-model.')
        except:
            print('Unable to load previous model for warm start. Continuing with random weights.')

    model.to(device)

    print('Neural network created')
    
    # ------  Loading datasets   ------
    
    dataset = RadicalsDataset(
        train=True,
        transform=CharacterTransform(),
        radical=rad,
    )

    test_dataset = RadicalsDataset(
        train=False,
        transform=CharacterTransform(),
        radical=rad,
    )
    
    print('Dataset created')

    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=6,
        sampler=ImbalancedDatasetSampler(
            dataset,
            callback_get_label=lambda dataset, idx: dataset.__getitem__(idx)['label'])
    )

    test_dataloader = data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=6
    )

    print('Dataloader created')
    
    optimizer = optim.Adam(params=model.parameters(), lr=0.0002)
   
    print('Optimizer created')
    
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)
    criterion = nn.BCEWithLogitsLoss()

    print('LR Scheduler created')

    # ------  Training loop   ------

    print('Starting training...')
    
    max_steps = 2200
    total_steps = 1

    for epoch in range(num_epochs):
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
                    accuracy = torch.sum(preds == classes).item() / batch_size
                    
                    try:
                        recall = torch.sum((preds == classes) * (classes == 1)).item() / torch.sum(classes == 1).item()
                        print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}, \tRec: {}'
                            .format(epoch + 1, total_steps, loss.item(), accuracy, recall))
                    except ZeroDivisionError:
                        print(f'No examples of {radical} in this batch.')
                        recall = 1
                        print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                            .format(epoch + 1, total_steps, loss.item(), accuracy))
                    
                    tbwriter.add_scalar('loss', loss.item(), total_steps)
                    tbwriter.add_scalar('accuracy', accuracy, total_steps)
                    tbwriter.add_scalar('recall', recall, total_steps)

            if total_steps % 700 == 0 or total_steps==1:
                with torch.no_grad():
                    print('Evaluating on validation set.')
                    
                    n_corr_classified = 0
                    n_images = 0
                    
                    n_corr_classified_and_class_instance = 0
                    n_class_instances = 0

                    for k, batch in enumerate(test_dataloader):
                        
                        if k % 2 == 0: # I treat batches with even indices as test set
                            continue

                        imgs = batch['img']
                        classes = batch['label']

                        imgs, classes = imgs.to(device), classes.to(device)
                        output = model(imgs)
                        classes = classes.type_as(output)
                        
                        preds = output > 0

                        n_corr_classified += torch.sum(preds == classes).item()
                        n_images += len(batch)

                        n_corr_classified_and_class_instance += torch.sum((preds == classes) * (classes == 1)).item()
                        n_class_instances += torch.sum(classes == 1).item()

                    try:
                        recall = n_corr_classified_and_class_instance / n_class_instances
                        accuracy = n_corr_classified / n_images

                    except ZeroDivisionError:
                        print(f'No examples of {radical} in testing data!')
                        recall = 1
                    
                    print('Validation accuracy:', accuracy, '\tValidation recall:', recall)

                    tbwriter.add_scalar('valid_loss', loss.item(), total_steps)
                    tbwriter.add_scalar('valid_accuracy', accuracy, total_steps)
                    tbwriter.add_scalar('valid_recall', recall, total_steps)
                        
                    if recall >= 0.8 and accuracy >= 0.85:
                        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model{radical}_at_state{total_steps}.pkl')
                        state = {
                            'test_set_results': evaluate_on_test_set(model, test_dataloader)
                            'epoch': epoch,
                            'total_steps': total_steps,
                            'optimizer': optimizer.state_dict(),
                            'model': model.state_dict()
                        }
                        torch.save(state, checkpoint_path)
                        return True
            
            total_steps += 1

        lr_scheduler.step()

        
    # ------ Saving model ------

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model{radical}_at_state{total_steps}.pkl')
    state = {
        'test_set_results': evaluate_on_test_set(model, test_dataloader)
        'epoch': epoch,
        'total_steps': total_steps,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict()
    }
    torch.save(state, checkpoint_path)
    return True


