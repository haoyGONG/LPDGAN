import torch
from models import networks
from models.networks import NLayerDiscriminator, PixelDiscriminator, SwinTransformer_Backbone
import functools
import sys
from collections import OrderedDict
import os
import torch.nn as nn
from models.config_su import get_config_or
sys.path.append("..")


class LPDGAN(nn.Module):
    def __init__(self, opt):
        super(LPDGAN, self).__init__()
        self.opt = opt
        self.mode = opt.mode
        self.gpu_ids = opt.gpu_ids
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        config_su = get_config_or()
        self.netG = SwinTransformer_Backbone(config_su).to(self.device)

        self.criterionL1 = torch.nn.L1Loss()
        self.perceptualLoss = networks.PerceptualLoss().to(self.device)

        if self.mode == 'train':
            self.model_names = ['G', 'D', 'D_smallblock', 'D1', 'D2']
            self.netD = NLayerDiscriminator(opt.input_nc + opt.output_nc, opt.ndf, n_layers=3, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)).to(self.device)
            self.netD1 = NLayerDiscriminator(opt.input_nc + opt.output_nc, opt.ndf, n_layers=3, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)).to(self.device)
            self.netD2 = NLayerDiscriminator(opt.input_nc + opt.output_nc, opt.ndf, n_layers=3, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)).to(self.device)

            self.netD_smallblock = PixelDiscriminator(opt.input_nc, opt.ndf, norm_layer=functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)).to(self.device)

            self.loss_names = ['G_GAN', 'G_L1', 'PlateNum_L1', 'D_GAN', 'P_loss', 'D_real', 'D_fake', 'D_s']
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionGAN_s = networks.GANLoss('lsgan').to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_D_smallblock = torch.optim.Adam(self.netD_smallblock.parameters(), lr=opt.lr,
                                                           betas=(0.5, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_D_smallblock)
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        else:

            self.model_names = ['G']
            self.loss_names = []


    def setup(self, opt):
        if self.mode == 'train':
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if self.mode != 'train' or opt.continue_train:
            load_suffix = opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_A1 = input['A1'].to(self.device)
        self.real_B1 = input['B1'].to(self.device)
        self.real_A2 = input['A2'].to(self.device)
        self.real_B2 = input['B2'].to(self.device)
        self.real_A3 = input['A3'].to(self.device)
        self.real_B3 = input['B3'].to(self.device)
        self.image_paths = input['A_paths']

        if self.mode == 'train':
            self.plate_info = input['plate_info'].to(self.device)

    def forward(self):
        self.fake_B, self.fake_B1, self.fake_B2, self.fake_B3, self.plate1, self.plate2 = self.netG(self.real_A,
                                                                                                    self.real_A1,
                                                                                                    self.real_A2,
                                                                                                    self.real_A3)
        self.fake_B_split = torch.chunk(self.fake_B, 7, dim=3)
        self.real_B_split = torch.chunk(self.real_B, 7, dim=3)
        self.real_A_split = torch.chunk(self.real_A, 7, dim=3)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def test(self):
        with torch.no_grad():
            self.forward()

    def cal_small_D(self):
        loss_D_s_fake = 0
        loss_D_s_real = 0
        for i in range(len(self.fake_B_split)):
            pred_s_fake = self.netD_smallblock(self.fake_B_split[i].detach())
            loss_D_s_fake_tmp = self.criterionGAN_s(pred_s_fake, False)

            pred_s_real = self.netD_smallblock(self.real_B_split[i].detach())
            loss_D_s_real_tmp = self.criterionGAN_s(pred_s_real, True)

            loss_D_s_fake += loss_D_s_fake_tmp
            loss_D_s_real += loss_D_s_real_tmp

        return loss_D_s_fake / 7.0, loss_D_s_real / 7.0

    def cal_small_G(self):
        loss_G_s_fake = 0
        for i in range(len(self.fake_B_split)):
            pred_s_fake = self.netD_smallblock(self.fake_B_split[i].detach())
            loss_G_s_fake_tmp = self.criterionGAN_s(pred_s_fake, True)

            loss_G_s_fake += loss_G_s_fake_tmp

        return loss_G_s_fake / 7.0

    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B),
                            1)
        pred_fake = self.netD(fake_AB.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        loss_D_real = self.criterionGAN(pred_real, True)

        fake_AB1 = torch.cat((self.real_A1, self.fake_B1),
                             1)
        pred_fake1 = self.netD1(fake_AB1.detach())
        loss_D_fake1 = self.criterionGAN(pred_fake1, False)

        real_AB1 = torch.cat((self.real_A1, self.real_B1), 1)
        pred_real1 = self.netD1(real_AB1)
        loss_D_real1 = self.criterionGAN(pred_real1, True)

        fake_AB2 = torch.cat((self.real_A2, self.fake_B2),
                             1)
        pred_fake2 = self.netD2(fake_AB2.detach())
        loss_D_fake2 = self.criterionGAN(pred_fake2, False)

        real_AB2 = torch.cat((self.real_A2, self.real_B2), 1)
        pred_real2 = self.netD2(real_AB2)
        loss_D_real2 = self.criterionGAN(pred_real2, True)

        self.loss_D_fake = (loss_D_fake + loss_D_fake1 + loss_D_fake2) / 3
        self.loss_D_real = (loss_D_real + loss_D_real1 + loss_D_real2) / 3

        self.loss_D_GAN = (loss_D_fake + loss_D_real + loss_D_fake1 + loss_D_real1 +
                           loss_D_fake2 + loss_D_real2) * 0.5 / 3

        loss_D_s_fake, loss_D_s_real = self.cal_small_D()
        self.loss_D_s = (loss_D_s_fake + loss_D_s_real) * 0.5

        self.loss_D_gp = (self.cal_gp(fake_AB, real_AB) + self.cal_gp(fake_AB1, real_AB1) +
                          self.cal_gp(fake_AB2, real_AB2)) * 10 / 3

        self.loss_D = self.loss_D_GAN + self.loss_D_gp + self.loss_D_s
        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        fake_AB1 = torch.cat((self.real_A1, self.fake_B1), 1)
        pred_fake1 = self.netD1(fake_AB1)
        loss_G_GAN1 = self.criterionGAN(pred_fake1, True)

        fake_AB2 = torch.cat((self.real_A2, self.fake_B2), 1)
        pred_fake2 = self.netD2(fake_AB2)
        loss_G_GAN2 = self.criterionGAN(pred_fake2, True)

        self.loss_G_GAN = (loss_G_GAN + loss_G_GAN1 + loss_G_GAN2) / 3

        self.loss_G_s = self.cal_small_G()

        loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        loss_G_L11 = self.criterionL1(self.fake_B1, self.real_B1) * self.opt.lambda_L1
        loss_G_L12 = self.criterionL1(self.fake_B2, self.real_B2) * self.opt.lambda_L1
        loss_G_L13 = self.criterionL1(self.fake_B3, self.real_B3) * self.opt.lambda_L1

        self.loss_G_L1 = (loss_G_L1 + loss_G_L11 + loss_G_L12 + loss_G_L13) / 4 * 0.01

        loss_P_loss = self.perceptualLoss(self.fake_B, self.real_B)
        loss_P_loss1 = self.perceptualLoss(self.fake_B1, self.real_B1)
        loss_P_loss2 = self.perceptualLoss(self.fake_B2, self.real_B2)
        loss_P_loss3 = self.perceptualLoss(self.fake_B3, self.real_B3)

        self.loss_P_loss = (loss_P_loss + loss_P_loss1 + loss_P_loss2 + loss_P_loss3) / 4 * 0.01

        self.loss_PlateNum_L1 = (self.criterionL1(self.plate1, self.plate_info) + self.criterionL1(self.plate2,
                                                                                                   self.plate_info)) / 2 * 0.01

        self.loss_G = self.loss_G_GAN + self.loss_G_s + self.loss_G_L1 + self.loss_P_loss + 0.1 * self.loss_PlateNum_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def cal_gp(self, fake_AB, real_AB):
        r = torch.rand(size=(real_AB.shape[0], 1, 1, 1))
        r = r.cuda()
        x = (r * real_AB + (1 - r) * fake_AB).requires_grad_(True)
        d = self.netD(x)
        fake = torch.ones_like(d)
        fake = fake.cuda()
        g = torch.autograd.grad(
            outputs=d,
            inputs=x,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True
        )[0]
        gp = ((g.norm(2, dim=1) - 1) ** 2).mean()
        return gp


    def get_image_paths(self):
        return self.image_paths

    def update_learning_rate(self):
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))
        return errors_ret

    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                torch.save(net.cpu().state_dict(), save_path)
                net.to(self.device)


    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                net.load_state_dict(state_dict)

def create_model(opt):
    return LPDGAN(opt=opt)

