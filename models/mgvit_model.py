import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from epdn import epdn_networks
from . import ipt
import math
from ECLoss.ECLoss import DCLoss
from TVLoss.TVLossL1 import TVLossL1 
from models.gradient import gradient
from torchvision import transforms
from models import common
class MGVIT(BaseModel):
    def name(self):
        return 'MGVIT'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.flip = transforms.RandomHorizontalFlip(p=1)

        self.gpu_ids
        #self.hl = opt.hl
        self.opt = opt
        

        # specify the training losses you want to print out. The program will call base_model.get_current_losses  loss_style_B + self.loss_content_A + self.loss_content_B
        self.loss_names = ['G','GAN', 'vgg', 'L1']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_A', 'real_B']


        self.visual_names = visual_names_A
        
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G','D']

        else:  # during test time, only load Gs
            self.model_names = ['G']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.conv = common.default_conv
        self.netG = ipt.define_G(opt, self.conv)
        

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            #print('input_nc', opt.input_nc, 'ndf', opt.ndf, 'which_model_netD',opt.which_model_netD, 'n_layers_D',opt.n_layers_D, 'norm',opt.norm, 'use_sigmoid',use_sigmoid, 'gpu_ids', self.gpu_ids)
            self.netD = networks.define_D(opt.input_nc*2, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)


        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()
            self.criterionVGG = epdn_networks.VGGLoss(self.gpu_ids)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device) # real world clear image
        self.real_B = input['B' if AtoB else 'A'].to(self.device) # real world hazy image

        self.image_paths = input['B_paths' if AtoB else 'B_paths']
        


    def forward(self):
        # Forward Cycle #
        # disentanglement #
        self.fake_A = self.netG(self.real_B)
        



    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        real_A_cat = torch.cat((self.real_B, self.real_A), dim=1)
        fake_A_cat = torch.cat((self.real_B, self.fake_A), dim=1)
        self.loss_D = self.backward_D_basic(self.netD, real_A_cat, fake_A_cat)


    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C = self.opt.lambda_content
        #lambda_S = self.opt.lambda_style
        


        self.loss_GAN = self.criterionGAN(self.netD(torch.cat((self.real_B, self.fake_A), dim=1)), True) * 0.0618

        
        # combined loss
        self.loss_G = self.loss_GAN


        # Vgg loss
        self.loss_vgg = self.criterionVGG(self.fake_A, self.real_A) * self.opt.lambda_vgg * 2#* 0.382
        
        self.loss_G += self.loss_vgg

        # Gradient loss
        
        self.gradient_real_A = gradient(self.real_A)
        self.gradient_fake_A = gradient(self.fake_A)
        

        self.loss_gradient_fake_A = 0.2 * self.criterionMSE(self.gradient_real_A, self.gradient_fake_A)
        
        self.loss_G += self.loss_gradient_fake_A

        self.loss_L1 = self.criterionIdt(self.real_A, self.fake_A)*3
        self.loss_G += self.loss_L1


        self.loss_G.backward()

    def optimize_parameters(self, opt):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad(self.netD, False)

        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        
        self.optimizer_D.step()
