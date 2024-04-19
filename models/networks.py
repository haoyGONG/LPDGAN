import torch
import torch.nn as nn
import functools
from torch.optim import lr_scheduler
import torchvision.models as models
import torchvision.transforms as transforms
from models.swin_transformer import SwinTransformerSys
import copy

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class GANLoss(nn.Module):

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        with torch.no_grad():
          self.vgg_relu_3_3 = self.contentFunc(15)
          self.vgg_relu_2_2 = self.contentFunc(8)
          self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def contentFunc(self, relu_layer):
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        model = model.eval()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == relu_layer:
                break

        return model

    def get_loss(self, fakeIm, realIm):

        fakeIm = self.transform(fakeIm)
        realIm = self.transform(realIm)

        f_fake_2_2 = self.vgg_relu_2_2.forward(fakeIm)
        f_real_2_2 = self.vgg_relu_2_2.forward(realIm)

        f_fake_3_3 = self.vgg_relu_3_3.forward(fakeIm)
        f_real_3_3 = self.vgg_relu_3_3.forward(realIm)

        f_real_2_2_no_grad = f_real_2_2.detach()
        f_real_3_3_no_grad = f_real_3_3.detach()
        mse = nn.MSELoss()
        loss = mse(f_fake_2_2, f_real_2_2_no_grad) + mse(f_fake_3_3, f_real_3_3_no_grad)
        return loss / 2.0

    def __call__(self, fakeIm, realIm):
        return self.get_loss(fakeIm, realIm)

class SwinTransformer_Backbone(nn.Module):
    def __init__(self, config, img_size=224, num_classes=3, zero_head=False, vis=False):
        super(SwinTransformer_Backbone, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x, x1, x2, x3):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        y, y1, y2, y3, p1, p2 = self.swin_unet(x, x1, x2, x3)
        return y, y1, y2, y3, p1, p2

    def unfreeze(self):
        for param in self.inception.parameters():
            param.requires_grad = True

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

        else:
            print("none pretrain")

class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)
