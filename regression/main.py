import math
import os
import click
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Echo import EchoSet
import sklearn.metrics
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from model import ViViT
import numpy as np
from utils import get_mean_and_std

@click.command("video")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default="/home/eda/Desktop/EE543-Term-Project/Video-Vision-Transformer/data")
@click.option("--output", type=click.Path(file_okay=False), default="output")
@click.option("--hyperparameter_dir", type=click.Path(file_okay=False), default="hyperparam_outputs")
@click.option("--run_test/--skip_test", default=True)
@click.option("--hyperparameter", type=bool, default=False)
@click.option("--num_epochs", type=int, default=35)
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=16)
@click.option("--num_heads", type=int, default=8)
@click.option("--num_layers", type=int, default=8)
@click.option("--projection_dim", type=int, default=2048)
@click.option("--device", type=str, default="cuda")
@click.option("--seed", type=int, default=0)

def run(

    data_dir,
    output,
    run_test,
    hyperparameter,
    hyperparameter_dir,
    num_epochs,
    lr,
    weight_decay,
    num_workers,
    batch_size,
    device,
    seed,
    projection_dim,
    num_heads,
    num_layers,
    input_shape  = (3, 128, 128, 128),
    patch_size   = (32, 32, 32),
):

    os.makedirs(output, exist_ok=True)  # Ensure the base output directory exists
    if hyperparameter:
        output = hyperparameter_dir
    output = os.path.join(output, f"lr_{lr}_wd_{weight_decay}_bs_{batch_size}_nh_{num_heads}_nl_{num_layers}_pd_{projection_dim}") if hyperparameter else output
    os.makedirs(output, exist_ok=True)

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

     # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
     # Initialize model, optimizer and criterion
    model       = ViViT(input_shape, patch_size, projection_dim, num_heads, num_layers)
    model       = model.to(device)
    # Set up optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Compute mean and std
    mean, std = get_mean_and_std(EchoSet(root=data_dir, split="train"))

    # Set up datasets and dataloaders
    dataset     = {}  
    # Load datasets
    dataset["train"]   = EchoSet(root=data_dir, split="train", mean=mean, std=std)
    dataset["val"]     = EchoSet(root=data_dir, split="val",  mean=mean, std=std)
    dataset["test"]    = EchoSet(root=data_dir, split="test", mean=mean, std=std)


    log_file_path = os.path.join(output, "log.csv")
    # Run training and testing loops
    with open(os.path.join(log_file_path), "a") as f:
        
        f.write("epoch,phase,loss,r2,time,y_size,batch_size\n")
        train_losses    = []
        val_losses      = []

        for epoch in range(num_epochs): 
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):  
                    torch.cuda.reset_peak_memory_stats(i)

                ds                                   = dataset[phase]
                dataloader                           = DataLoader(ds, batch_size=batch_size, shuffle=True,num_workers=num_workers)
                loss, yhat, y, filename, repeat, fps = run_epoch(model, dataloader, phase, optimizer, device,train_losses,val_losses)
                
                f.write("{},{},{},{},{},{},{}\n".format(
                                                            epoch,
                                                            phase,
                                                            loss,
                                                            sklearn.metrics.r2_score(y, yhat),
                                                            time.time() - start_time,
                                                            y.size,
                                                            batch_size))
                f.flush()
            scheduler.step(epoch + 1)

        if run_test:
            split = "test"
            ds = dataset[split]
            
            dataloader                          = DataLoader(EchoSet(root=data_dir, split=split, mean=mean, std=std),batch_size=1, 
                                                 shuffle=True,num_workers=num_workers)
            loss, yhat, y, filename,repeat, fps = run_epoch(model, dataloader, split, optimizer, device,train_losses=[],val_losses=[])
            # Write full performance to file
            with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
                g.write("filename,true_value, prediction,repeat, fps\n")
                for (file, pred, target, repeat, fps) in zip(filename, yhat, y, repeat, fps):
                        g.write("{},{},{:.4f},{},{}\n".format(file,target,pred, repeat, fps))

                g.write("{} R2:   {:.3f} \n".format(split, sklearn.metrics.r2_score(y, yhat)))
                g.write("{} MAE:  {:.2f} \n".format(split, sklearn.metrics.mean_absolute_error(y, yhat)))
                g.write("{} RMSE: {:.2f} \n".format(split, math.sqrt(sklearn.metrics.mean_squared_error(y, yhat))))
                g.flush()

    
    np.save(os.path.join(output, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(output, "val_losses.npy"), np.array(val_losses))
    print(f"Train and validation losses saved to {output}")

def run_epoch(model, dataloader, phase, optimizer,device,train_losses, val_losses):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        
    """
    model.train(phase== "train")

    yhat = []          # Prediction 
    y    = []          # Ground truth

    n    = 0           # number of videos processed
    s1   = 0           # sum of ground truth EF
    s2   = 0           # Sum of ground truth EF squared

    train_loss = 0.0
    with torch.set_grad_enabled(phase== "train"):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (filename, video, ejection, repeat, fps) in dataloader:

                video, ejection = video.float().to(device), ejection.float().to(device)
                video           = video.permute(0, 2, 1, 3, 4)  # [Batch, Channel, Depth, Height, Width]

                y.append(ejection.cpu().numpy())

                
                s1              += ejection.sum()              # Mean * n 
                s2              += (ejection ** 2).sum()       # Varience * n
                
                outputs         = model(video)
                yhat.append(outputs.view(-1).to("cpu").detach().numpy())

                loss            = torch.nn.functional.mse_loss(outputs.view(-1), ejection)
                if phase== "train":
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                    optimizer.step()  
                
                
                train_loss  += loss.item() * video.size(0)
                n           += video.size(0)
                pbar.set_postfix_str("{:.2f} ({:.2f}) / {:.2f}".format(train_loss / n, loss.item(), s2 / n - (s1 / n) ** 2))
                pbar.update(1)

                if (phase== "train"):
                    train_losses.append(train_loss/ n)
                elif phase== "val":
                    val_losses.append(train_loss/ n)
                else:
                    pass
                   
    yhat    = np.concatenate(yhat)
    y       = np.concatenate(y)

    return train_loss / n, yhat, y, filename, repeat, fps

if __name__ == "__main__":
    run()
