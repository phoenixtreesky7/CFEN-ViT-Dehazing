### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch

def create_pretrained_model(opt, which_premodel):
    if opt.pretrained_model == 'pix2pixHD':
    	from .pix2pixHD_model import Pix2PixHDModel
    	model = Pix2PixHDModel()
    else:
    	from .ui_model import UIModel
    	model = UIModel()
    model.initialize(opt, which_premodel)
    print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
