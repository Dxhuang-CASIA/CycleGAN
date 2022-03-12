import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module): # 不改变大小
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        # [B, C, H, W]
        conv_block = [nn.ReflectionPad2d(1), # [B, C, H + 2, W + 2]
                      nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1), # [B, C, H, W]
                      nn.InstanceNorm2d(in_channels),
                      nn.ReLU(inplace = True),
                      nn.ReflectionPad2d(1), # [B, C, H + 2, W + 2]
                      nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1), # [B, C, H, W]
                      nn.InstanceNorm2d(in_channels)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks = 9):
        super(Generator, self).__init__()
        # c7s1-64
        model = [nn.ReflectionPad2d(3), # [B, 3, 262, 262]
                 nn.Conv2d(input_nc, 64, kernel_size = 7, stride = 1), # [B, 64, 256, 256]
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace = True)]

        # d128,d256 [B, 64, 256, 256] => [B, 256, 64, 64]
        in_channels = 64
        out_channels = in_channels * 2
        for _ in range(2):
            model += [nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1),
                      nn.InstanceNorm2d(out_channels),
                      nn.ReLU(inplace = True)]
            in_channels = out_channels
            out_channels = in_channels * 2

        # R256 * 9 [B, 256, 64, 64] => [B, 256, 64, 64]
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_channels)]

        # u128,u64 [B, 256, 64, 64] => [B, 128, 130, 120] => [B, 64, 262, 262]
        out_channels = in_channels // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3, stride = 2, output_padding = 1),
                      nn.InstanceNorm2d(out_channels),
                      nn.ReLU(inplace = True)]
            in_channels = out_channels
            out_channels = in_channels // 2

        # c7s1-3 [B, 64, 262, 262] => [B, 3, 256, 256]
        model += [nn.Conv2d(in_channels, output_nc, kernel_size = 7, stride = 1),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Dicriminator(nn.Module):
    def __init__(self, input_nc):
        super(Dicriminator, self).__init__()
        # C64-C128-C256-C512
        model = [nn.Conv2d(input_nc, 64, kernel_size = 4, stride = 2, padding = 1),
                 nn.InstanceNorm2d(64),
                 nn.LeakyReLU(0.2, inplace = True)]

        model += [nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace = True)]

        model += [nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace = True)]

        model += [nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace = True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, kernel_size = 4, padding = 1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

# model = Generator(3, 3)
# x = torch.rand(1, 3, 256, 256)
# model2 = Dicriminator(3)
# print(model2(model(x)).shape)