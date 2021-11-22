import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from epdn import epdn_networks
from . import dec_ipt
import math
from ECLoss.ECLoss import DCLoss
from TVLoss.TVLossL1 import TVLossL1 
from models.gradient import gradient
from torchvision import transforms
from models import common
import pytorch_msssim
class DECRHLGVIT(BaseModel):
    def name(self):
        return 'DECRLGVIT'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.flip = transforms.RandomHorizontalFlip(p=1)

        self.gpu_ids
        #self.hl = opt.hl
        self.opt = opt
        

        # specify the training losses you want to print out. The program will call base_model.get_current_losses  loss_style_B + self.loss_content_A + self.loss_content_B
        self.loss_names = ['G', 'GAN_a', 'GAN_r', 'vgg_a', 'vgg_r', 'L2_a', 'L2_r', 'gradient_fake_a', 'gradient_fake_r', 'ssim_a', 'ssim_r']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_A', 'real_B', 'fake_R', 'real_R']


        self.visual_names = visual_names_A
        
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G','D_A', 'D_R']

        else:  # during test time, only load Gs
            self.model_names = ['G']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.conv = common.default_conv
        if opt.model_G == 'iidr_hlgvit_crs_gd4':
            print('This is the iidr_hlgvit_crs_gd4 model')
            from . import networks_iidr_hlgvit_crs_gd4
            self.netG = networks_iidr_hlgvit_crs_gd4.define_G(opt, self.conv)
        
        

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            #print('input_nc', opt.input_nc, 'ndf', opt.ndf, 'which_model_netD',opt.which_model_netD, 'n_layers_D',opt.n_layers_D, 'norm',opt.norm, 'use_sigmoid',use_sigmoid, 'gpu_ids', self.gpu_ids)
            self.netD_A = networks.define_D(opt.input_nc*2, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_R = networks.define_D(opt.input_nc*2, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            #self.netD_S = networks.define_D(opt.input_nc*2, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)


        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_R_pool = ImagePool(opt.pool_size)
            #self.fake_S_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionSSIM = pytorch_msssim.SSIM()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()
            self.criterionVGG = epdn_networks.VGGLoss(self.gpu_ids)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_R.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device) # real world clear image
        self.real_B = input['B' if AtoB else 'A'].to(self.device) # real world hazy image
        self.real_R = input['R'].to(self.device)
        #self.real_S = input['S'].to(self.device)

        self.image_paths = input['B_paths' if AtoB else 'B_paths']
        


    def forward(self):
        # Forward Cycle #
        # disentanglement #
        [self.fake_R, self.fake_A] = self.netG(self.real_B)
        
        


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
        self.loss_DA = self.backward_D_basic(self.netD_A, real_A_cat, fake_A_cat)

        fake_R = self.fake_R_pool.query(self.fake_R)
        real_R_cat = torch.cat((self.real_B, self.real_R), dim=1)
        fake_R_cat = torch.cat((self.real_B, self.fake_R), dim=1)
        self.loss_DR = self.backward_D_basic(self.netD_R, real_R_cat, fake_R_cat)

        


    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C = self.opt.lambda_content
        #lambda_S = self.opt.lambda_style        

        # GAN loss

        self.loss_GAN_a = self.criterionGAN(self.netD_A(torch.cat((self.real_B, self.fake_A), dim=1)), True) * 0.0618
        self.loss_GAN_r = self.criterionGAN(self.netD_R(torch.cat((self.real_B, self.fake_R), dim=1)), True) * 0.0618
        #self.loss_GAN_s = self.criterionGAN(self.netD_S(torch.cat((self.real_B, self.fake_S), dim=1)), True) * 0.0618
        #self.loss_GAN_pa = self.criterionGAN(self.netD_A(torch.cat((self.real_B, self.fake_PA), dim=1)), True) * 0.0618
        self.loss_GAN = self.loss_GAN_a + self.loss_GAN_r #+ self.loss_GAN_s #+ self.loss_GAN_pa


        # Vgg loss

        self.loss_vgg_a = self.criterionVGG(self.fake_A, self.real_A) * self.opt.lambda_vgg * 2#* 0.382
        self.loss_vgg_r = self.criterionVGG(self.fake_R, self.real_R) * self.opt.lambda_vgg * 2#* 0.382
        #self.loss_vgg_s = self.criterionVGG(self.fake_S, self.real_S) * self.opt.lambda_vgg * 2#* 0.382
        #self.loss_vgg_pa = self.criterionVGG(self.fake_PA, self.real_A) * self.opt.lambda_vgg * 2#* 0.382
        self.loss_vgg = self.loss_vgg_a + self.loss_vgg_r# + self.loss_vgg_s #+ self.loss_vgg_pa

        # Gradient loss
        
        self.gradient_real_A = gradient(self.real_A)
        self.gradient_fake_A = gradient(self.fake_A)
        #self.gradient_fake_PA = gradient(self.fake_PA)
        self.gradient_real_R = gradient(self.real_R)
        self.gradient_fake_R = gradient(self.fake_R)
        #self.gradient_real_S = gradient(self.real_S)
        #self.gradient_fake_S = gradient(self.fake_S)

        self.loss_gradient_fake_a = self.criterionMSE(self.gradient_real_A, self.gradient_fake_A) * 2
        self.loss_gradient_fake_r = self.criterionMSE(self.gradient_real_R, self.gradient_fake_R) * 2
        #self.loss_gradient_fake_s = self.criterionMSE(self.gradient_real_S, self.gradient_fake_S) * 2
        #self.loss_gradient_fake_pa = self.criterionMSE(self.gradient_real_A, self.gradient_fake_PA) * 2
        self.loss_gradient = self.loss_gradient_fake_a + self.loss_gradient_fake_r #+ self.loss_gradient_fake_s #+ self.loss_gradient_fake_pa 

        # L2 Loss

        self.loss_L2_a = self.criterionIdt(self.real_A, self.fake_A) * 2
        self.loss_L2_r = self.criterionIdt(self.real_R, self.fake_R) * 2
        #self.loss_L2_s = self.criterionIdt(self.real_S, self.fake_S) * 2
        #self.loss_L2_pa = self.criterionIdt(self.real_A, self.fake_PA) * 2
        self.loss_L2 = self.loss_L2_a + self.loss_L2_r #+ self.loss_L2_s #+ self.loss_L2_pa

        # SSIM Loss

        self.loss_ssim_a = (1-self.criterionSSIM(self.real_A, self.fake_A)) * 3
        self.loss_ssim_r = (1-self.criterionSSIM(self.real_R, self.fake_R)) * 3
        #self.loss_ssim_s = (1-self.criterionSSIM(self.real_S, self.fake_S)) * 3
        #self.loss_ssim_pa = (1-self.criterionSSIM(self.real_S, self.fake_PA)) * 1
        self.loss_ssim = self.loss_ssim_a + self.loss_ssim_r #+ self.loss_ssim_s #+ self.loss_ssim_pa

        

        self.loss_G = self.loss_GAN + self.loss_vgg + self.loss_gradient + self.loss_L2 + self.loss_ssim


        self.loss_G.backward()

    def optimize_parameters(self, opt):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A,self.netD_R], False)

        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A,self.netD_R], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        
        self.optimizer_D.step()
