"""
Youtu Lab; tencent; China
Joefzhou.
Model CDSR:
patch encoder + infonce + memory bank + dynamic + RRDB + DQA
"""

from matplotlib.pyplot import new_figure_manager
import torch
import torch.nn.functional as F
from torch import nn
import sys
sys.path.append('..')
import model.common as common
from moco.builder import MoCo
import functools
from collections import OrderedDict
import model.module_util as mutil


def make_model(args):
    return BlindSR(args)


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# -------------------------------------------------
# SFT layer and RRDB block for non-blind RRDB-SFT
# -------------------------------------------------



class CrossModalAttention(nn.Module):
    """ CMA attention Layer"""

    def __init__(self, in_dim, activation=None, ratio=8, cross_value=True):
        super(CrossModalAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.cross_value = cross_value

        self.query_conv = nn.Linear(in_dim, in_dim, bias=True)
        self.key_conv = nn.Linear(in_dim, in_dim, bias=True)
        self.value_conv = nn.Linear(in_dim, in_dim, bias=True)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        B, C = x.size()

        proj_query = self.query_conv(x).unsqueeze(dim=2)  # (B, C, 1)
        proj_key = self.key_conv(y).unsqueeze(dim=1)  # (B, 1, C)
        energy = torch.bmm(proj_query, proj_key)  # B, C, C
        attention = self.softmax(energy)  # BX (C) X (C)

        proj_value = self.value_conv(y).unsqueeze(dim=1)   # B , C

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).squeeze()

        out = self.gamma * out + y

        if self.activation is not None:
            out = self.activation(out)

        return out  # , attention

    def flops(self, n):
        # calculate flops for 1 window with token length of n
        flops = 0
        # qkv = self.qkv(x)
        # flops += n * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.chanel_in * self.chanel_in
        #  x = (attn @ v)
        flops += self.chanel_in * self.chanel_in
        # x = self.proj(x)
        return flops


class DQA(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DQA, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.sf_attn = CrossModalAttention(channels_in)
        self.kernel = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64 * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = common.default_conv(channels_in, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()
        feat_ave = torch.nn.functional.adaptive_avg_pool2d(x[0], 1).view(b, c)
        kernel_embedding = self.sf_attn(feat_ave, x[1])  # （B C)
        # branch 1
        kernel = self.kernel(kernel_embedding).view(-1, 1, self.kernel_size, self.kernel_size)  # (64*b, 1, 3, 3)
        out = self.relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b * c,
                                 padding=(self.kernel_size - 1) // 2))  # in (1, b*c, H, W) -> (1, b*c, H, W)
        out = self.conv(out.view(b, -1, h, w))

        # branch 2
        out = out + self.ca(x)

        return out

class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in // reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        att = self.conv_du(x[1][:, :, None, None])

        return x[0] * att

class FuseBLK(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction):
        super(FuseBLK, self).__init__()

        self.dqa1 = DQA(n_feat, n_feat, kernel_size, reduction)
        self.dqa2 = DQA(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size, stride=1, padding=(kernel_size//2))
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size, stride=1, padding=(kernel_size//2))

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''

        out = self.relu(self.dqa1(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.dqa2([out, x[1]]))
        out = self.conv2(out) + x[0]

        return out

class SFT_Layer(nn.Module):
    ''' SFT layer '''
    def __init__(self, nf=64, para=256):
        super(SFT_Layer, self).__init__()
        self.fc = nn.Sequential(
					nn.Linear(para, 256),
					nn.PReLU(),
					nn.Linear(256, nf),
					nn.Sigmoid()
				)
        self.fuse = FuseBLK(nf, kernel_size=3, reduction=2)

    def forward(self, feature_maps, para_maps):
        para_maps = self.fc(para_maps)
        x = self.fuse([feature_maps, para_maps])
        return x


class ResidualDenseBlock_5C(nn.Module):
    '''  Residual Dense Block '''
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB_SFT(nn.Module):
    ''' Residual in Residual Dense Block with SFT layer '''

    def __init__(self, nf, gc=32, para=256):
        super(RRDB_SFT, self).__init__()
        self.SFT = SFT_Layer(nf=nf, para=para)
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, input):
        out = self.SFT(input[0], input[1])
        out = self.RDB1(out)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return [out * 0.2 + input[0], input[1]]

class ActConvWithBN(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, leky=0.1):
        super(ActConvWithBN, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(leky)

    def forward(self, x):
        return self.act(self.bn1(self.conv1(x)))

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # pixel wise
        self.conv1 = ActConvWithBN(3,    64, kernel_size=7, stride=1, padding=3)
        self.conv2 = ActConvWithBN(64,   64, kernel_size=1, stride=1, padding=0)
        self.conv3 = ActConvWithBN(64,  128, kernel_size=3, stride=2, padding=1)
        self.conv4 = ActConvWithBN(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv5 = ActConvWithBN(128, 256, kernel_size=7, stride=1, padding=3)
        self.conv6 = ActConvWithBN(256, 256, kernel_size=1, stride=1, padding=0)
        
        # patch based
        self.pconv1 = ActConvWithBN(3,    64, kernel_size=12, stride=12, padding=0)  # b, c, h/ps, w/ps
        self.pconv2 = ActConvWithBN(64,   64, kernel_size=3, stride=1, padding=1)
        self.pconv3 = ActConvWithBN(64,  128, kernel_size=3, stride=1, padding=1)
        self.pconv4 = ActConvWithBN(128, 256, kernel_size=3, stride=1, padding=1)

        self.ave_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp1 = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

        # codebook
        self.feature_bank = nn.Embedding(1024, 256)
        self.fc_q = nn.Linear(256, 256)
        self.fc_k = nn.Linear(256, 256)
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        x1 = self.conv5(x1)
        x1 = self.conv6(x1)
        x1_ave = self.ave_pool(x1).squeeze(-1).squeeze(-1)

        x2 = self.pconv1(x)
        x2 = self.pconv2(x2)
        x2 = self.pconv3(x2)
        x2 = self.pconv4(x2)
        x2_ave = self.ave_pool(x2).squeeze(-1).squeeze(-1)

        # codebook
        fea = self.mlp1(torch.cat([x1_ave, x2_ave], dim=1))
        q = self.fc_q(fea)
        fb = self.feature_bank.weight
        k = self.fc_k(fb)
        qk = torch.mm(q, k.transpose(1, 0))
        qk = F.softmax(qk, dim=1)
        fea = torch.mm(qk, fb)
        out = self.mlp(fea)

        return fea, out


class CDSR(nn.Module):
    def __init__(self, args):
        super(CDSR, self).__init__()
        in_nc = 3
        out_nc = 3
        nf = 64
        gc = 32
        scale = int(args.scale[0])
        kernel_size = 21
        nb = 10
        embedding_length = 256

        self.scale = scale
        self.kernel_size = kernel_size

        self.compress = nn.Sequential(
			nn.Linear(embedding_length, 256),
			nn.PReLU(),
			nn.Linear(256, 256),
			nn.PReLU(),
			nn.Linear(256, 256),
			nn.PReLU(),
			nn.Linear(256, 256),
			nn.PReLU()
		)

        RRDB_SFT_block_f = functools.partial(RRDB_SFT, nf=nf, gc=gc, para=256)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = mutil.make_layer(RRDB_SFT_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upsampler = sequential(nn.Conv2d(nf, out_nc * (scale ** 2), kernel_size=3, stride=1, padding=1, bias=True),
                                    nn.PixelShuffle(scale))

    def forward(self, x, fea):
        # paddingBottom = int(np.ceil(h / self.ps) * self.ps - h)
        # paddingRight = int(np.ceil(w / self.ps) * self.ps - w)
        # x = torch.nn.functional.pad(x, [0, paddingRight, 0, paddingBottom], mode='reflect')

        fea_map = self.compress(fea)
        lr_fea = self.conv_first(x)
        fea = self.RRDB_trunk([lr_fea, fea_map])
        fea = lr_fea + self.trunk_conv(fea[0])
        out = self.upsampler(fea)
        return out


class BlindSR(nn.Module):
    def __init__(self, args):
        super(BlindSR, self).__init__()

        # Generator
        self.G = CDSR(args)

        # Encoder
        self.E = MoCo(base_encoder=Encoder)

    def get_construct_learning_output(self, x_query, x_key):
        _, output, target = self.E(im_q=x_query, im_k=x_key)
        return output, target

    def freeze_encoder(self):
        for lpara in self.E.encoder_q.parameters():
            lpara.requires_grad = False
        for lpara in self.E.encoder_k.parameters():
            lpara.requires_grad = False

    def forward(self, x):
        # self.freeze_encoder()
        x = x / 255.0
        if self.training:
            x_query = x[:, 0, ...]  # b, c, h, w
            x_key = x[:, 1, ...]  # b, c, h, w

            # degradation-aware represenetion learning
            fea, logits, labels = self.E(x_query, x_key)

            # degradation-aware SR
            sr = self.G(x_query, fea)
            sr = sr * 255.0
            return sr, logits, labels
        else:
            # degradation-aware represenetion learning
            fea = self.E(x, x)

            # degradation-aware SR
            sr = self.G(x, fea)
            sr = sr * 255.0

            return sr


if __name__ == '__main__':
    from option import args

    args.scale = [2]
    model = BlindSR(args)
    model.eval()
    # x = torch.rand(1, 3, 48, 48)
    # y = model(x)

    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True,
                                              print_per_layer_stat=True)  # 不用写batch_size大小，默认batch_size=1
    print('Flops:  ' + flops)
    print('Params: ' + params)
