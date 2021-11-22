import subprocess
import sys
import time
import os
sys.path.append("D:/dzhao/DHViT/DHViT")
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import torch
import logging
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

#torch.cuda.set_device(1)

if __name__ == '__main__':
	opt = TrainOptions().parse()
	data_loader = CreateDataLoader(opt)
	dataset = data_loader.load_data()
	dataset_size = len(data_loader)
	logging.info('#training images = %d' % dataset_size)
	model = create_model(opt)
	#model.load_state_dict(checkpoint['state_dict'], strict=False)
	model.setup(opt)
	visualizer = Visualizer(opt)
	total_steps = 0
	for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
		if epoch != 1:
			time.sleep(160)
		epoch_start_time = time.time()
		iter_data_time = time.time()
		epoch_iter = 0
		opt.current_epoch = epoch
		logging.info("Current epoch update to {}".format(opt.current_epoch))
		for i, data in enumerate(dataset):
			if total_steps == 0:
				for item in data.items():
					if isinstance(item[1], torch.Tensor):
						logging.info(item[1].size())
			iter_start_time = time.time()
			if total_steps % opt.print_freq == 0:
				t_data = iter_start_time - iter_data_time
			visualizer.reset()
			total_steps += opt.batchSize
			epoch_iter += opt.batchSize
			model.set_input(data)
			model.optimize_parameters(opt)
			
			if total_steps % opt.display_freq == 0:
				save_result = total_steps % opt.update_html_freq == 0
				visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
			
			if total_steps % opt.print_freq == 0:
				losses = model.get_current_losses()
				t = (time.time() - iter_start_time) / opt.batchSize
				visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
				if opt.display_id > 0:
					visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)
			if total_steps % 1000 == 0:
				print('begin time sleep')
				time.sleep(36)
				print('stop time sleep')

			if total_steps % opt.save_latest_freq == 0:
				logging.info('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
				model.save_networks('latest')
			iter_data_time = time.time()
		
		if epoch % opt.save_epoch_freq == 0:
			logging.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
			model.save_networks('latest')
			model.save_networks(epoch)
		
		logging.info('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.max_epoch, time.time() - epoch_start_time))
		model.update_learning_rate()
