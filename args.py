import argparse

def train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type = int, default = 0, help = 'starting epoch')
    parser.add_argument('--n_epochs', type = int, default = 200, help = 'epoch')
    parser.add_argument('--batchSize', type = int, default = 1, help = 'batchsize')
    parser.add_argument('--dataroot', type = str, default = './datasets/horse2zebra/', help = 'root directory of the dataset')
    parser.add_argument('--lr', type = float, default = 0.0002, help = 'initial learning rate')
    parser.add_argument('--decay_epoch', type = int, default = 100, help = 'decay learning rate epoch')
    parser.add_argument('--imgsize', type = int, default = 256, help = 'image size')
    parser.add_argument('--input_nc', type = int, default = 3, help = 'input data channel')
    parser.add_argument('--output_nc', type = int, default = 3, help = 'output data channel')
    parser.add_argument('--cuda', type = bool, default= True,  help = 'use GPU computation')
    parser.add_argument('--n_cpu', type = int, default = 0, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--outf', type = str, default = './img_sys', help = '中间生成图片')
    parser.add_argument('--ckpt', type = str, default = './ckpt', help = 'save checkpoints')
    args = parser.parse_args()
    return args

def test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type = int, default = 1, help = 'batchsize')
    parser.add_argument('--dataroot', type = str, default = './datasets/horse2zebra/', help = 'root directory of the dataset')
    parser.add_argument('--input_nc', type = int, default = 3, help = 'input data channel')
    parser.add_argument('--output_nc', type = int, default = 3, help = 'output data channel')
    parser.add_argument('--imgsize', type = int, default = 256, help = 'image size')
    parser.add_argument('--cuda', type = bool, default = True, help = 'use GPU computation')
    parser.add_argument('--n_cpu', type = int, default = 0, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type = str, default = './ckpt/netG_A2B_epoch_199.pth', help = 'A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type = str, default = './ckpt/netG_B2A_epoch_199.pth', help = 'B2A generator checkpoint file')
    parser.add_argument('--outputA', type = str, default = './output_img/A')
    parser.add_argument('--outputB', type = str, default = './output_img/B')
    args = parser.parse_args()
    return args