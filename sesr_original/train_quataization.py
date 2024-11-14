import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import numpy as np
import os
from tqdm import tqdm
import random
import torch.quantization as quant
import wandb

from dataset import LoadDataset
from helpers import evaluate_average_psnr as psnr
from utils import *

# Seed everything for reproducibility
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# Initialize model and prepare for quantization
def prepare_model_for_quantization(model):
    # Fuse layers for quantization (if needed for your model's architecture)
    model_fused = torch.quantization.fuse_modules(model, [['conv1', 'relu'], ['conv2', 'relu2']])
    
    # Set the quantization configuration for quantization-aware training
    model_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Prepare the model for QAT (Quantization Aware Training)
    torch.quantization.prepare_qat(model_fused, inplace=True)
    
    return model_fused


# Save both regular and quantized models
def save_checkpoint(epoch, model, model_fused, optimizer, scheduler, best_psnr, checkpoint_save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_psnr': best_psnr,
        'random_seed': args.random_seed
    }, checkpoint_save_path)

    # Save the quantized model checkpoint
    checkpoint_save_path_quant = checkpoint_save_path.replace(".pth", "_quantized.pth")
    torch.quantization.convert(model_fused, inplace=True)
    torch.save(model_fused.state_dict(), checkpoint_save_path_quant)
    print(f'Quantized model saved to {checkpoint_save_path_quant}')


# Training loop
def train(epoch, model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss = 0.0
    psnr_values = []

    for i, (inputs, targets) in enumerate(tqdm(train_loader, desc='Training')):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        psnr_value = psnr(outputs, targets)
        psnr_values.append(psnr_value.item())

    avg_train_loss = running_loss / len(train_loader)
    average_psnr = np.mean(psnr_values)
    return avg_train_loss, average_psnr


# Validation loop
def validate(epoch, model, val_loader, device):
    model.eval()
    psnr_values = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validation'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            psnr_value = psnr(outputs, targets)
            psnr_values.append(psnr_value.item())

    average_psnr = np.mean(psnr_values)
    return average_psnr


def main():
    global best_psnr, fn_name
    epochs = args.epoch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    if args.model == 'm3':
        model = SESRRelease_M3(scaling_factor=args.scaling_factor)
    elif args.model == 'm5':
        model = SESRRelease_M5(scaling_factor=args.scaling_factor)
    elif args.model == 'm7':
        model = SESRRelease_M7(scaling_factor=args.scaling_factor)
    elif args.model == 'm11':
        model = SESRRelease_M11(scaling_factor=args.scaling_factor)

    model = model.to(device)
    model_fused = prepare_model_for_quantization(model)

    # Setup optimizer, criterion, and scheduler
    optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999))
    criterion = nn.L1Loss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8000, gamma=0.5)

    # Load datasets
    train_dataset = LoadDataset(train_hr_images, args.scaling_factor, transform=train_transform)
    val_dataset = LoadDataset(val_hr_images, args.scaling_factor, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    best_psnr = 0

    for epoch in range(epochs):
        train_loss, train_psnr = train(epoch + 1, model, optimizer, criterion, train_loader, device)
        val_psnr = validate(epoch + 1, model, val_loader, device)

        # Save checkpoint
        if epoch % args.checkpoint_save == 0:
            checkpoint_save_path = f'/database/iyj0121/ckpt/sesr/FRB/sesr_{fn_name}_{epoch}.pth'
            save_checkpoint(epoch, model, model_fused, optimizer, scheduler, best_psnr, checkpoint_save_path)

        # Update and save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            checkpoint_save_path = f'/database/iyj0121/ckpt/sesr/FRB/sesr_{fn_name}_best.pth'
            save_checkpoint(epoch, model, model_fused, optimizer, scheduler, best_psnr, checkpoint_save_path)
            print(f'Best model saved to {checkpoint_save_path}')

        wandb.log({'train loss': train_loss, 'train psnr': train_psnr, 'val psnr': val_psnr}, step=epoch)

    # Save the last model
    last_model_save_path = f'./checkpoint/sesr_{fn_name}_last.pth'
    save_checkpoint(epochs, model, model_fused, optimizer, scheduler, best_psnr, last_model_save_path)
    print(f'Last model saved to {last_model_save_path}')


if __name__ == '__main__':
    main()
