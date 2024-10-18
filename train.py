import os
import torch
from data import train_dataloader
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn


def FFT_loss(pred_img, label_img):
	criterion = nn.L1Loss()

	label_fft = torch.rfft(label_img)
	pred_fft = torch.rfft(pred_img)
	
	loss = criterion(pred_fft, label_fft)
	
	return loss

def _train(model, config):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	criterion = torch.nn.L1Loss()
	optimizer = torch.optim.AdamW(model.parameters(),
								  lr=config.learning_rate,
								  weight_decay=config.weight_decay,
								  betas=(0.9, 0.9))
	
	dataloader = train_dataloader(config.data_dir, config.batch_size, config.num_worker)
	max_iter = len(dataloader)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)
	
	writer_main_loss = SummaryWriter(r'./runs/main_loss')
	writer_aux_loss = SummaryWriter(r'./runs/aux_loss')
	writer = SummaryWriter(r'./runs')
	epoch_timer = Timer('m')
	iter_timer = Timer('m')
	
	model_save_overwrite = os.path.join(config.model_save_dir, 'model_overwrite.pkl')
	if os.path.isfile(model_save_overwrite) and config.Resume == True:
		state_dict = torch.load(model_save_overwrite)
		model.load_state_dict(state_dict['model'])
		optimizer.load_state_dict(state_dict["optimizer"])
		scheduler.load_state_dict(state_dict["scheduler"])
		start_epoch = state_dict['epoch']
		lr = check_lr(optimizer)
		print("\n Model Restored, epoch = %4d\n" % (start_epoch + 1))
		print("\n                 current lr = %10f\n" % (lr))
	else:
		print("No previous data... Started from scratch ... \n")
		start_epoch = 0
	
	best_psnr = -1
	for epoch_idx in range(start_epoch + 1, config.num_epoch + 1):
		epoch_timer.tic()
		iter_timer.tic()

		for iter_idx, batch_data in enumerate(tqdm(dataloader)):
			
			input_img, label_img = batch_data
			input_img = input_img.to(device)
			label_img = label_img.to(device)
			
			optimizer.zero_grad()
			pred_img, visual = model(input_img)
			
			main_loss = criterion(pred_img, label_img)
			aux_loss = 0.1 * FFT_loss(pred_img, label_img)
			
			loss = main_loss + aux_loss
			
			loss.backward()
			optimizer.step()
			scheduler.step()
			
			if (iter_idx + 1) % config.print_freq == 0:
				lr = check_lr(optimizer)

		if epoch_idx % config.save_freq == 0:
			torch.save({'model': model.state_dict(),
						'optimizer': optimizer.state_dict(),
						'scheduler': scheduler.state_dict(),
						'epoch': epoch_idx}, model_save_overwrite)

		if epoch_idx % config.valid_freq == 0:
			val = _valid(model, config, epoch_idx)
			print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
			writer.add_scalar('PSNR', val, epoch_idx)
			
			if val >= best_psnr:
				torch.save({'model': model.state_dict()}, os.path.join(config.model_save_dir, 'Best.pkl'))
			
			save_name = os.path.join(config.model_save_dir, 'model_%d.pkl' % epoch_idx)
			torch.save({'model': model.state_dict(),
						'optimizer': optimizer.state_dict(),
						'scheduler': scheduler.state_dict(),
						'epoch': epoch_idx}, save_name)
	
	save_name = os.path.join(config.model_save_dir, 'Final.pkl')
	torch.save({'model': model.state_dict()}, save_name)
