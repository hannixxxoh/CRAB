import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import argparse
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
from PIL import Image
from torchsummary import summary
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imresize import *
from dataset import LoadDataset
from helpers import evaluate_average_psnr as psnr
from utils import *

import wandb
import random



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# 현재 스크립트 파일의 디렉토리 경로를 가져옴
script_dir = os.path.dirname(os.path.abspath(__file__))
# 모델 파일이 있는 디렉토리의 경로
model_dir = os.path.join(script_dir, 'model')

# 모델 파일들이 있는 디렉토리를 파이썬의 sys.path에 추가
sys.path.append(model_dir)
torch.set_num_threads(1)

from sesr_models import *
import albumentations as A

# 훈련 데이터 변환 설정
# train_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(), 
#     transforms.RandomVerticalFlip(),  
#     transforms.RandomRotation(90),  
#     transforms.RandomCrop(256)
# ])

transform_list = [
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0),  # 90도 단위로 회전
    A.RandomCrop(height=256, width=256, p=1.0)  # 256x256 크롭
]
train_transform = A.Compose(transform_list)

# 명령행 인자 파서 설정
parser = argparse.ArgumentParser(description='Train QuickSRNet models.')
parser.add_argument('--model', type=str, choices=['m3', 'm5', 'm7', 'm11', 'clip'], required=True,
                    help='Size of the model to train: m3, m5, m7, m11')
parser.add_argument('--scaling_factor', type=int, default=2, 
                    help='Scaling factor for super-resolution')
parser.add_argument('--epoch', type=int, default=20000)
parser.add_argument('--checkpoint_save', type=int, default=1000)
parser.add_argument('--checkpoint_load_path', type=str, default=None)
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument("--clipped_relu",action="store_true")
parser.add_argument("--concat_residual",action="store_true")
parser.add_argument("--df2k",action="store_true")
parser.add_argument("--scheduler",type=str, default='ConstantLR')

args = parser.parse_args()


# 데이터셋 디렉토리 설정
if args.df2k:
    train_hr_dir = "/database/hanni/SR_benchmark/DF2K/HR"
else:
    train_hr_dir = "/database/iyj0121/dataset/DIV2K/DIV2K_train_HR/"
val_hr_dir = "/database/iyj0121/dataset/DIV2K/DIV2K_valid_HR/"

# 훈련 및 검증 이미지 경로 리스트 생성
train_hr_images = sorted([os.path.join(train_hr_dir, img) for img in os.listdir(train_hr_dir)])
val_hr_images = sorted([os.path.join(val_hr_dir, img) for img in os.listdir(val_hr_dir)])

# 데이터셋 및 데이터로더 생성
train_dataset = LoadDataset(train_hr_images, args.scaling_factor, transform=train_transform)
val_dataset = LoadDataset(val_hr_images, args.scaling_factor, transform=None)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

# 모델 선택
if args.model == 'm3':
    model = SESRRelease_M3(scaling_factor=args.scaling_factor, clipped_relu=args.clipped_relu, concat_residual=args.concat_residual)
elif args.model == 'm5':
    model = SESRRelease_M5(scaling_factor=args.scaling_factor, clipped_relu=args.clipped_relu, concat_residual=args.concat_residual)
elif args.model == 'm7':
    model = SESRRelease_M7(scaling_factor=args.scaling_factor, clipped_relu=args.clipped_relu, concat_residual=args.concat_residual)
elif args.model == 'm11':
    model = SESRRelease_M11(scaling_factor=args.scaling_factor, clipped_relu=args.clipped_relu, concat_residual=args.concat_residual)
elif args.model == 'clip':
    model = CLIPSR(num_blocks=[3, 4, 6, 3])

# 모델 파라미터 수 확인
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"# params (model_fp32): {num_params}")

# GPU가 사용 가능한지 확인하고 모델을 GPU로 이동
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
for layer in model.children():
    print(layer)

# 손실 함수 및 옵티마이저 정의
criterion = nn.L1Loss()
fn_name = ''

# 학습률 스케줄러
if 'step' in args.scheduler.lower():
    optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8000, gamma=0.5)
# elif 'cosine' in args.scheduler.lower():
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=25e-5)
elif 'cosinewarmup' in args.scheduler.lower():
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer,
                                              T_0=args.epoch,
                                              eta_max=5e-4,
                                              T_up=10)
elif 'constant' in args.scheduler.lower():
    optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=1)
else:
    raise ValueError(f"Invalid scheduler: {args.scheduler}")
    

# 훈련 루프 파라미터
best_psnr = 0  # 최적의 PSNR 초기화

use_clipped_relu_str = "clipped_relu" if args.clipped_relu else "p_relu"
use_concat_str = "concat" if args.concat_residual else "add"
use_df2k_str = "DF2K" if args.df2k else "DIV2K"

# 훈련 함수
def train(epoch):

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

        if epoch % args.checkpoint_save == 0:
            if not os.path.exists('/database/iyj0121/ckpt/clip_cliped_relu'):
                os.makedirs('/database/iyj0121/ckpt/clip_cliped_relu')
            checkpoint_save_path = f'/database/iyj0121/ckpt/clip_cliped_relu/clip_{fn_name}_{epoch}.pth'
            save_checkpoint(epoch, checkpoint_save_path)
        
    
    avg_train_loss = running_loss / len(train_loader)
    average_psnr = np.mean(psnr_values)
    lr = optimizer.param_groups[0]['lr']
    print(f'Epoch [{epoch}/{args.epoch}], LR {lr}, Loss: {avg_train_loss:.4f}, Average PSNR on train set: {average_psnr:.3f} dB')
    wandb.log({'train loss': avg_train_loss,
               'train psnr': average_psnr,
               'lr': lr},
               step=epoch)
    if scheduler is not None:
        scheduler.step()
    return avg_train_loss, average_psnr

# 체크포인트 저장 함수
def save_checkpoint(epoch, checkpoint_save_path):
    # checkpoint_save_path = f'./checkpoint/sesr_checkpoint_{args.model}_scale_{args.scaling_factor}.pth'
    # print(f'Checkpoint model saved at epoch {epoch}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_psnr': best_psnr,
        'random_seed': args.random_seed
    }, checkpoint_save_path)


# validation 함수
def validate(epoch):
    global best_psnr
    model.eval()
    psnr_values = []
    outputs_list = []
    output_dir = f'./output'  # Output 저장 폴더 설정
    
    with torch.no_grad():
        for j, (inputs, targets) in enumerate(tqdm(val_loader, desc='Validation')):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # outputs_list.append(outputs)
            psnr_value = psnr(outputs, targets)
            psnr_values.append(psnr_value.item())
            
            # # 이미지를 PIL 이미지로 변환하여 저장
            # output_image = transforms.ToPILImage()(outputs.squeeze().cpu())
            # output_image.save(os.path.join(output_dir, f'output_{j}.png'))

    average_psnr = np.mean(psnr_values)
    print(f'Epoch [{epoch}/{args.epoch}], Average PSNR on validation set: {average_psnr:.3f} dB')

    # Check if current PSNR is the best record
    if average_psnr > best_psnr:
        best_psnr = average_psnr
        if not os.path.exists('/database/iyj0121/ckpt/clip_cliped_relu/'):
            os.makedirs('/database/iyj0121/ckpt/clip_cliped_relu/')
        checkpoint_save_path = f'/database/iyj0121/ckpt/clip_cliped_relu/clip_{fn_name}_best.pth'
        save_checkpoint(epoch, checkpoint_save_path)
        # torch.save(model.state_dict(), checkpoint_save_path)
        print(f'Best model saved to {checkpoint_save_path}')
    wandb.log({'best psnr': best_psnr,
               'epoch':epoch,
               'val psnr': average_psnr},
               step=epoch)
    return average_psnr

# Main function
def main():
    global best_psnr
    global fn_name
    epochs = args.epoch

    if args.random_seed is not None:
        print(f'Random seed: {args.random_seed}')
        seed_everything(args.random_seed)

    # 이전 체크포인트가 있는지 확인
    #if os.path.exists(args.checkpoint_load_path):
    if args.checkpoint_load_path is not None and os.path.exists(args.checkpoint_load_path):
        # 이전 체크포인트를 불러오고 이어서 훈련을 시작
        checkpoint = torch.load(args.checkpoint_load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        checkpoint_epoch = checkpoint['epoch']
        best_psnr = checkpoint['best_psnr']
        print(f'Checkpoint loaded from epoch {checkpoint_epoch}')
        print(f'Continue training from epoch {checkpoint_epoch}')
        print("Best PSNR: ", best_psnr)

        if 'random_seed' in checkpoint.keys():
            print("random seed:", checkpoint['random_seed'])
            args.random_seed = checkpoint['random_seed']
            seed_everything(args.random_seed)
        
        fn_name = f'{use_concat_str}_{use_clipped_relu_str}_{args.model}_{args.scaling_factor}_Random{args.random_seed}_{args.scheduler}_Rot90_Crop256_{use_df2k_str}'


        config_dict = {
            'model': args.model,
            'scaling_factor': args.scaling_factor,
            'random_seed': args.random_seed,
            'use_clipped_relu': args.clipped_relu,
            'use_concat': args.concat_residual,
            'scheduler': args.scheduler,
            'randomrotation': [90, 180, 270, 360],
            'randomcrop': 256,
            'dataset': use_df2k_str
        }

        wandb.init(project='clip',
                name=f'clip_cliped_relu_{fn_name}_baseline',
                entity='iyj01',
                config=config_dict)

        for epoch in range(checkpoint_epoch-1, epochs):
            train_loss, train_psnr = train(epoch+1)
            val_psnr = validate(epoch+1)
    else:
        print('Checkpoint not found. Starting training from scratch.')
        
        if args.random_seed is None:
            args.random_seed = random.randint(0, 10000)
            print(f'Random seed: {args.random_seed}')
            seed_everything(args.random_seed)

        fn_name = f'{use_concat_str}_{use_clipped_relu_str}_{args.model}_{args.scaling_factor}_Random{args.random_seed}_{args.scheduler}_Rot90_Crop256_{use_df2k_str}'
        config_dict = {
            'model': args.model,
            'scaling_factor': args.scaling_factor,
            'random_seed': args.random_seed,
            'use_clipped_relu': args.clipped_relu,
            'use_concat': args.concat_residual,
            'scheduler': args.scheduler,
            'randomrotation': [90, 180, 270, 360],
            'randomcrop': 256,
            'dataset': use_df2k_str
        }

        wandb.init(project='clip',
                name=f'clip_cliped_relu_{fn_name}_baseline',
                entity='iyj01',
                config=config_dict)
    
        for epoch in range(epochs):

            train_loss, train_psnr = train(epoch+1)
            val_psnr = validate(epoch+1)

        
    # 훈련 종료시 모델 저장
    last_model_save_path = f'./checkpoint/sesr_{fn_name}_last.pth'
    torch.save(model.state_dict(), last_model_save_path)
    print(f'Last model saved to {last_model_save_path}')
    

if __name__ == '__main__':
    main()