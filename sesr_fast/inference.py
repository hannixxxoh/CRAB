import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from PIL import Image
from dataset import *
from helpers import evaluate_average_psnr as psnr_fn
from helpers import evaluate_average_lpips as lpips_fn

from sesr_models import *


# inference without clipping
def inference(model, data_loader, output_dir, device):

    model.eval()
    psnr_values = []
    with torch.no_grad():
        for idx, (img_name, inputs, targets) in enumerate(tqdm(data_loader, desc='Inference')):
            inputs, targets = inputs.to(device), targets.to(device)
            img_name = img_name[0]
            outputs = model(inputs)
            psnr_value = psnr_fn(outputs, targets)
            psnr_values.append(psnr_value.item())
            # 결과 이미지 저장 (Optional)
            output_img = outputs.squeeze().cpu().numpy().transpose(1, 2, 0) * 255
            output_img = Image.fromarray(output_img.astype('uint8'))
            output_img.save(os.path.join(output_dir, f'{img_name}.png'))

    average_psnr = np.mean(psnr_values)
    print(f'Average PSNR on validation set: {average_psnr:.2f} dB')

# inference with clipping
def clip_inference(model, data_loader, output_dir, device):

    model.eval()
    psnr_values = []
    lpips_values = []
    with torch.no_grad():
        for idx, (img_name, inputs, targets) in enumerate(tqdm(data_loader, desc='Inference')):
            inputs, targets = inputs.to(device), targets.to(device)
            img_name = img_name[0]
            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0, 1)
            psnr_value = psnr_fn(outputs, targets)
            lpips_value = lpips_fn(outputs, targets)

            psnr_values.append(psnr_value.item())
            lpips_values.append(lpips_value.item())
            
            # 결과 이미지 저장 (Optional)
            output_img = outputs.squeeze().cpu().numpy().transpose(1, 2, 0) * 255
            output_img = Image.fromarray(output_img.astype('uint8'))
            output_img.save(os.path.join(output_dir, f'{img_name}.png'))

    average_lpips = np.mean(lpips_values)
    average_psnr = np.mean(psnr_values)
    print(f'Average LPIPS on validation set: {average_lpips:.4f} dB')
    print(f'Average PSNR on validation set: {average_psnr:.2f} dB')

def main():
    
    hr_dirs = ["/database/hanni/DIV2KS/DIV2K_valid_HR"]
    model_path = "/home/hanni/project/aimet-model-zoo/Pytorch/code/sesr_concat_clip/checkpoint/sesr_add_clipped_relu_m7_2_Random32_ConstantLR_Rot90_Crop256_DIV2K_best.pth" # 모델 경로
    scaling_factor = 2  # scaling factor
    clipped_relu = True  # clipped relu 사용 여부
    concat_residual = False  # concat residual 사용 여부
    model_size = 'm7'

    # Model selection
    if model_size == 'm3':
        model = SESRRelease_M3(scaling_factor=scaling_factor, clipped_relu=clipped_relu, concat_residual=concat_residual)
    elif model_size == 'm5':
        model = SESRRelease_M5(scaling_factor=scaling_factor, clipped_relu=clipped_relu, concat_residual=concat_residual)
    elif model_size == 'm7':
        model = SESRRelease_M7(scaling_factor=scaling_factor, clipped_relu=clipped_relu, concat_residual=concat_residual)
    elif model_size == 'm11':
        model = SESRRelease_M11(scaling_factor=scaling_factor, clipped_relu=clipped_relu, concat_residual=concat_residual)
    else:
        raise ValueError("Invalid model size. Choose 'm3', 'm5', 'm7', or 'm11'.")

    # GPU 사용 가능 여부 확인 및 모델 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # 모델 로드
    # Trained
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint['epoch']
    best_psnr = checkpoint['best_psnr']

    model.collapse()
    for layer in model.children():
        print(layer)
    print("Model Size:", model_size)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"# params (model_fp32): {round(num_params/1000, 2)}")
    print("epoch: ", epoch)
    print("best_psnr: ", best_psnr)
    # 결과 이미지를 저장할 디렉토리 생성
    for hr_dir in hr_dirs:
        print("\n")
        print(hr_dir)
        
        # hr_dir에서 이미지 경로를 가져옴
        hr_images = sorted([os.path.join(hr_dir, img) for img in os.listdir(hr_dir) if "4Kto8K" not in img])


        # 데이터셋 준비
        # val_transform = transforms.Compose([])  # 검증 세트에는 변환을 적용하지 않음
        dataset = LoadFastValDataset(hr_images, scaling_factor, transform=None)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        output_dir = os.path.join('./output', os.path.splitext(os.path.basename(model_path))[0], hr_dir.split('/')[-1] if "LX" in hr_dir else hr_dir.split('/')[-2])
        os.makedirs(output_dir, exist_ok=True)
        
        # 추론
        clip_inference(model, data_loader, output_dir, device)

if __name__=="__main__":
    main()
