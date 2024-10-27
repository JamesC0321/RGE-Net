import os

import PIL.Image
import cv2
import torch
from torchvision.transforms import functional as F1
import numpy as np
import torch.nn.functional as F
from data import test_dataloader
import time
import sys, scipy.io

def _eval(model, config):
	model_pretrained = os.path.join('results/', config.model_name, 'weights/', config.test_model)
	state_dict = torch.load(model_pretrained)
	model.load_state_dict(state_dict['model'])
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	dataloader = test_dataloader(config.data_dir, batch_size=1, num_workers=0)
	dataloader_len = dataloader.__len__()
	
	model.eval()
	
	with torch.no_grad():


		for iter_idx, data in enumerate(dataloader):
			input_img, label_img, _ = data
			input_img = input_img.to(device)
			tm = time.time()
			_ = model(input_img)
			_ = time.time() - tm

			if iter_idx == 20:
				break
		
		
		for iter_idx, data in enumerate(dataloader):
			input_img, label_img, fn = data
			input_img = input_img.to(device)
			label_img = label_img.to(device)

			torch.cuda.synchronize()
			tm = time.time()
			pred, _ = model(input_img)
			elaps = time.time() - tm

			p_numpy = pred.squeeze(0).cpu().numpy()
			p_numpy = np.clip(p_numpy, 0, 1)
			in_numpy = label_img.squeeze(0).cpu().numpy()

			save_path = os.path.join(config.result_dir, 'save_path')
			save_name = os.path.join(save_path, rf'{fn[0]}')
			pred_clip = torch.clamp(pred, 0, 1)
			pred_clip += 0.5 / 255
			pred = F1.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
			pred.save(save_name)

			print('%d / %d iter' % (iter_idx + 1, dataloader_len))
			print(elaps)