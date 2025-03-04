import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import numpy as np
import os
from torch.autograd import Variable

def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss



class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg19', VGG19())
        self.criterion = torch.nn.L1Loss(reduction="mean")
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg19(x), self.vgg19(y)

        p1 = self.weights[0] * \
            self.criterion(x_vgg['relu1_2'], y_vgg['relu1_2'])
        p2 = self.weights[1] * \
            self.criterion(x_vgg['relu2_2'], y_vgg['relu2_2'])
        p3 = self.weights[2] * \
            self.criterion(x_vgg['relu3_2'], y_vgg['relu3_2'])
        p4 = self.weights[3] * \
            self.criterion(x_vgg['relu4_2'], y_vgg['relu4_2'])
        p5 = self.weights[4] * \
            self.criterion(x_vgg['relu5_2'], y_vgg['relu5_2'])

        return p1+p2+p3+p4+p5


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        # https://pytorch.org/hub/pytorch_vision_vgg/
        mean = np.array(
            [0.485, 0.456, 0.406], dtype=np.float32)
        mean = mean.reshape((1, 3, 1, 1))
        self.mean = Variable(torch.from_numpy(mean)).cuda()
        std = np.array(
            [0.229, 0.224, 0.225], dtype=np.float32)
        std = std.reshape((1, 3, 1, 1))
        self.std = Variable(torch.from_numpy(std)).cuda()
        self.initial_model()

    def forward(self, x):
        relu1_1 = self.relu1_1((x-self.mean)/self.std)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out

    def load_pretrained(self, vgg19_weights_path, gpu):
        if os.path.exists(vgg19_weights_path):
            if torch.cuda.is_available():
                data = torch.load(vgg19_weights_path)
                print("load vgg_pretrained_model:"+vgg19_weights_path)
            else:
                data = torch.load(vgg19_weights_path,
                                  map_location=lambda storage, loc: storage)
            self.initial_model(data)
            self.to(gpu)
        else:
            print("you need download vgg_pretrained_model in the directory of  "+str(self.config.DATA_ROOT) +
                  "\n'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'")
            raise Exception("Don't load vgg_pretrained_model")

    def initial_model(self,data=None):
            vgg19 = models.vgg19()
            if data is not None:
                vgg19.load_state_dict(data)
            features = vgg19.features
            self.relu1_1 = torch.nn.Sequential()
            self.relu1_2 = torch.nn.Sequential()

            self.relu2_1 = torch.nn.Sequential()
            self.relu2_2 = torch.nn.Sequential()

            self.relu3_1 = torch.nn.Sequential()
            self.relu3_2 = torch.nn.Sequential()
            self.relu3_3 = torch.nn.Sequential()
            self.relu3_4 = torch.nn.Sequential()

            self.relu4_1 = torch.nn.Sequential()
            self.relu4_2 = torch.nn.Sequential()
            self.relu4_3 = torch.nn.Sequential()
            self.relu4_4 = torch.nn.Sequential()

            self.relu5_1 = torch.nn.Sequential()
            self.relu5_2 = torch.nn.Sequential()
            self.relu5_3 = torch.nn.Sequential()
            self.relu5_4 = torch.nn.Sequential()

            for x in range(2):
                self.relu1_1.add_module(str(x), features[x])

            for x in range(2, 4):
                self.relu1_2.add_module(str(x), features[x])

            for x in range(4, 7):
                self.relu2_1.add_module(str(x), features[x])

            for x in range(7, 9):
                self.relu2_2.add_module(str(x), features[x])

            for x in range(9, 12):
                self.relu3_1.add_module(str(x), features[x])

            for x in range(12, 14):
                self.relu3_2.add_module(str(x), features[x])

            for x in range(14, 16):
                self.relu3_3.add_module(str(x), features[x])

            for x in range(16, 18):
                self.relu3_4.add_module(str(x), features[x])

            for x in range(18, 21):
                self.relu4_1.add_module(str(x), features[x])

            for x in range(21, 23):
                self.relu4_2.add_module(str(x), features[x])

            for x in range(23, 25):
                self.relu4_3.add_module(str(x), features[x])

            for x in range(25, 27):
                self.relu4_4.add_module(str(x), features[x])

            for x in range(27, 30):
                self.relu5_1.add_module(str(x), features[x])

            for x in range(30, 32):
                self.relu5_2.add_module(str(x), features[x])

            for x in range(32, 34):
                self.relu5_3.add_module(str(x), features[x])

            for x in range(34, 36):
                self.relu5_4.add_module(str(x), features[x])

            # don't need the gradients, just want the features
            # for param in self.parameters():
            #     param.requires_grad = False
if __name__ == "__main__":
    input = torch.rand([1,3,256,256])
    loss = VGGPerceptualLoss()
    l = loss(input, input)

