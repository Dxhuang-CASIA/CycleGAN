import os
import torch
from model import Generator
from args import test_args
import torchvision.utils as vutils
from datasets import ImageDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

def test(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG_A2B = Generator(args.input_nc, args.output_nc).to(device)
    netG_B2A = Generator(args.output_nc, args.input_nc).to(device)
    netG_A2B.load_state_dict(torch.load(args.generator_A2B))
    netG_B2A.load_state_dict(torch.load(args.generator_B2A))

    netG_A2B.eval()
    netG_B2A.eval()

    Tensor = torch.cuda.FloatTensor if args.cuda else torch.Tensor
    input_A = Tensor(args.batchSize, args.input_nc, args.imgsize, args.imgsize)
    input_B = Tensor(args.batchSize, args.output_nc, args.imgsize, args.imgsize)

    transforms_ = [transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    datasets = ImageDataset(args.dataroot, transforms_ = transforms_, mode = 'test')
    dataloader = DataLoader(datasets, batch_size = args.batchSize, shuffle = False, num_workers = args.n_cpu)

    if not os.path.exists(args.outputA):
        os.makedirs(args.outputA)
    if not os.path.exists(args.outputB):
        os.makedirs(args.outputB)

    for step, batch in enumerate(dataloader):
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
        fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

        vutils.save_image(fake_A, '%s/%04d.png' % (args.outputA, step + 1))
        vutils.save_image(fake_B, '%s/%04d.png' % (args.outputB, step + 1))

if __name__ == '__main__':
    args = test_args()
    test(args)