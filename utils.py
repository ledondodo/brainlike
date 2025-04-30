### Utils
import h5py
import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import explained_variance_score
from scipy.stats import pearsonr
from typing import List, Callable, Dict

import torch


def load_it_data(path_to_data):
    """ Load IT data

    Args:
        path_to_data (str): Path to the data

    Returns:
        np.array (x6): Stimulus train/val/test; objects list train/val/test; spikes train/val
    """

    datafile = h5py.File(os.path.join(path_to_data,'IT_data.h5'), 'r')

    stimulus_train = datafile['stimulus_train'][()]
    spikes_train = datafile['spikes_train'][()]
    objects_train = datafile['object_train'][()]
    
    stimulus_val = datafile['stimulus_val'][()]
    spikes_val = datafile['spikes_val'][()]
    objects_val = datafile['object_val'][()]
    
    stimulus_test = datafile['stimulus_test'][()]
    objects_test = datafile['object_test'][()]

    ### Decode back object type to latin
    objects_train = [obj_tmp.decode("latin-1") for obj_tmp in objects_train]
    objects_val = [obj_tmp.decode("latin-1") for obj_tmp in objects_val]
    objects_test = [obj_tmp.decode("latin-1") for obj_tmp in objects_test]

    return stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val


def visualize_img(stimulus,objects,stim_idx):
    """Visualize image given the stimulus and corresponding index and the object name.

    Args:
        stimulus (array of float): Stimulus containing all the images
        objects (list of str): Object list containing all the names
        stim_idx (int): Index of the stimulus to plot
    """    
    normalize_mean=[0.485, 0.456, 0.406]
    normalize_std=[0.229, 0.224, 0.225]

    img_tmp = np.transpose(stimulus[stim_idx],[1,2,0])

    ### Go back from normalization
    img_tmp = (img_tmp*normalize_std + normalize_mean) * 255

    plt.figure()
    plt.imshow(img_tmp.astype(np.uint8),cmap='gray')
    plt.title(str(objects[stim_idx]))
    plt.show()
    return

def evaluation_metrics(predictions, spikes, keys: List[str]=None):
    """Plot and print evaluation metrics (explained variance and correlation) for the predictions in the given keys.
    This function will create a histogram for each metric and each key, and print the mean explained variance and correlation.
    In the histogram, the explained variance and correlation are plotted on the x-axis, and the number of neurons is plotted on the y-axis.
    Args:
        predictions (dict): Dictionary of predictions
        spikes (np.ndarray): True spike values
        keys (list): List of keys to evaluate (must be keys in predictions dict)
    """
    if keys is None:
        keys = list(predictions.keys())
    nplots = len(keys)
    fig, axs = plt.subplots(nplots, 2, figsize=(10, 5), squeeze=False, sharex='col', sharey='col')

    # For each prediction in keys
    predictions = {
        k: v for k,v in predictions.items()
        if k in keys
    }
    for idx, (key, pred) in enumerate(predictions.items()):
        ev = explained_variance_score(spikes, pred, multioutput='raw_values')
        corr = np.array([pearsonr(x,y)[0] for (x,y) in zip(pred.T, spikes.T)])
        print(f"-- {key} --")
        print("Mean EV:", ev.mean())
        print("Mean Pearson correlation:", corr.mean())

        # Explained Variance
        axs[idx, 0].hist(ev, bins=30, range=(-3,1.2), edgecolor='black', alpha=0.7)
        axs[idx, 0].set_title(f'{key} - Explained Variance Score')
        axs[idx, 0].set_xlabel('Explained Variance')
        axs[idx, 0].set_ylabel('Number of Neurons')

        # Correlation
        axs[idx, 1].hist(corr, bins=30, range=(-0.2,1.2), edgecolor='black', alpha=0.7)
        axs[idx, 1].set_title(f'{key} - Correlation')
        axs[idx, 1].set_xlabel('Correlation Coefficient')
        axs[idx, 1].set_ylabel('Number of Neurons')

    plt.tight_layout()
    plt.show()

def compare_metrics(metric_fn: Callable[[np.ndarray, np.ndarray], float],
                    y_true: np.ndarray,
                    predictions: Dict[str, np.ndarray],
                    key_before: str,
                    key_after: str,
                    metric_name: str) -> None:
    """Compare metrics before and after a change in the model.
    Args:
        metric_fn (Callable): Function to compute the metric
        y_true (np.ndarray): True spike values
        predictions (Dict[str, np.ndarray]): Dictionary of predictions
        key_before (str): Key for the predictions before the change
        key_after (str): Key for the predictions after the change
        metric_name (str): Name of the metric
    """
    before = metric_fn(y_true, predictions[key_before])
    after = metric_fn(y_true, predictions[key_after])
    print(f"{metric_name}: after {after} - before {before}")
    print(f"{metric_name} residual {after - before}\n")

def train_model(model, loss_fn, opt, train_loader, val_inputs, val_targets, metric_fn, epochs):
    """
    Train the model using the provided loss function and optimizer.
    Args:
        model (torch.nn.Module): The model to train.
        loss_fn (callable): The loss function to use.
        opt (torch.optim.Optimizer): The optimizer to use.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        metric_fn (callable): The metric function to use for evaluation.
        epochs (int): Number of epochs to train for.
    """
    for ep in range(epochs):
        # Training
        model.train()
        for it, batch in enumerate(train_loader):
            inputs, targets = batch

            # Run forward pass
            predictions = model(inputs)
            
            # Compute loss
            loss = loss_fn(predictions, targets)
            
            # Run backward pass
            loss.backward()
            
            # Update the weights using optimizer
            opt.step()
            
            # Zero-out the accumualted gradients
            opt.zero_grad()

            print('\rEp {}/{}, it {}/{}: loss train: {:.2f}'.
                  format(ep + 1, epochs, it + 1, len(train_loader), loss), end='')

        # Validation
        model.eval()
        with torch.no_grad():
            metric = metric_fn(val_targets.detach().cpu().numpy(), model(val_inputs).detach().cpu().numpy())
            print(', metric validation: {:.2f}'.format(metric))