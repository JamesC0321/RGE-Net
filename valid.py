import torch
from torchvision.transforms import functional as F
from data import valid_dataloader, test_dataloader
import os
import numpy as np
import sys, scipy.io

from torchvision.transforms import Resize

def _valid(model, config, ep):
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = valid_dataloader(config.data_dir, batch_size=1, num_workers=0)
    model.eval()

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            input_img, label_img,fn = data
            input_img = input_img.to(device)

            pred,_ = model(input_img)

            p_numpy = pred.squeeze(0).cpu().numpy()
            p_numpy = np.clip(p_numpy, 0, 1)
            in_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, in_numpy, data_range=1)

    print('\n')
    model.train()
    return psnr

