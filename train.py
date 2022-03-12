import torch
import torch.optim as optim
from PIL import Image
from datasets import ImageDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.utils as vutils
import itertools
import matplotlib.pyplot as plt
from utils import count_parameters, weights_init_normal, LambdaLR, ReplayBuffer
from args import train_args
from model import Generator, Dicriminator

def train(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    netG_A2B = Generator(args.input_nc, args.output_nc).to(device)
    netG_B2A = Generator(args.output_nc, args.input_nc).to(device)
    netD_A = Dicriminator(args.input_nc).to(device)
    netD_B = Dicriminator(args.output_nc).to(device)
    print('the number of parameters of the model:', count_parameters(
        netG_A2B) + count_parameters(netG_B2A) + count_parameters(netD_A) + count_parameters(netD_B))

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_Cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & Schedulers
    optimizer_G = optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr = args.lr, betas = (0.5, 0.999))
    optimizer_D_A = optim.Adam(netD_A.parameters(), lr = args.lr, betas = (0.5, 0.999))
    optimizer_D_B = optim.Adam(netD_B.parameters(), lr = args.lr, betas = (0.5, 0.999))

    lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G,
                                                 lr_lambda = LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                         lr_lambda = LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                         lr_lambda = LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)

    # 输入目标内存分配
    Tensor = torch.cuda.FloatTensor if args.cuda else torch.Tensor
    input_A = Tensor(args.batchSize, args.input_nc, args.imgsize, args.imgsize)
    input_B = Tensor(args.batchSize, args.output_nc, args.imgsize, args.imgsize)
    target_real = Variable(Tensor(args.batchSize).fill_(1.0), requires_grad = False)
    target_fake = Variable(Tensor(args.batchSize).fill_(0.0), requires_grad = False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    transforms_ = [transforms.Resize(int(args.imgsize * 1.12), Image.BICUBIC), # 放大插值
                   transforms.RandomCrop(args.imgsize),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    dataset = ImageDataset(args.dataroot, transforms_ = transforms_, unaligned = True)
    dataloader = DataLoader(dataset, batch_size = args.batchSize, shuffle = True,
                            num_workers = args.n_cpu)

    loss_G_list = []
    loss_D_list = []
    loss_G_GAN_list = []
    loss_G_identity_list = []
    loss_G_cycle_list = []

    for epoch in range(args.epoch, args.n_epochs):
        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()
        for step, batch in enumerate(dataloader):
            # Set input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))

            ######## Generators A2B & B2A ########
            optimizer_G.zero_grad()

            # Identity loss
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B) * 5.0

            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * 5.0

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_Cycle_ABA = criterion_Cycle(recovered_A, real_A) * 10.0

            recovered_B = netG_A2B(fake_A)
            loss_Cycle_BAB = criterion_Cycle(recovered_B, real_B) * 10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_Cycle_ABA + loss_Cycle_BAB

            loss_G.backward()
            optimizer_G.step()

            ######## Discriminator A ########
            optimizer_D_A.zero_grad()

            # real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5

            loss_D_A.backward()
            optimizer_D_A.step()

            ######## Discriminator B ########
            optimizer_D_B.zero_grad()

            # real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5

            loss_D_B.backward()
            optimizer_D_B.step()

            # 存数据

            loss_G_list.append(loss_G.item())
            loss_G_identity_list.append((loss_identity_A + loss_identity_B).item())
            loss_G_GAN_list.append((loss_GAN_A2B + loss_GAN_B2A).item())
            loss_G_cycle_list.append((loss_Cycle_ABA + loss_Cycle_BAB).item())
            loss_D_list.append((loss_D_A + loss_D_B).item())

            if step % 5 == 0:
                print('Epoch:%d\titer:%d\tloss_G:%.4f\tloss_G_identity:%.4f\tloss_G_GAN:%.4f\tloss_G_Cycle:%.4f\tloss_D:%.4f'
                      %(epoch, step, loss_G_list[-1],loss_G_identity_list[-1],
                        loss_G_GAN_list[-1],loss_G_cycle_list[-1], loss_D_list[-1]))
                vutils.save_image(real_A, '%s/realA_epoch_%03d.png' % (args.outf, epoch), normalize=True)
                vutils.save_image(real_B, '%s/realB_epoch_%03d.png' % (args.outf, epoch), normalize=True)
                vutils.save_image(fake_B.detach(), '%s/fakeB_epoch_%03d.png' % (args.outf, epoch), normalize=True)
                vutils.save_image(fake_A.detach(), '%s/fakeA_epoch_%03d.png' % (args.outf, epoch), normalize=True)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            torch.save(netG_A2B.state_dict(), '%s/netG_A2B_epoch_%d.pth' % (args.ckpt, epoch))
            torch.save(netG_B2A.state_dict(), '%s/netG_B2A_epoch_%d.pth' % (args.ckpt, epoch))
            torch.save(netD_A.state_dict(), '%s/netD_A_epoch_%d.pth' % (args.ckpt, epoch))
            torch.save(netD_B.state_dict(), '%s/netD_B_epoch_%d.pth' % (args.ckpt, epoch))

        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

    plt.figure(figsize = (10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(loss_G_list, label = "loss_G")
    plt.plot(loss_G_identity_list, label = "loss_G_identity")
    plt.plot(loss_G_GAN_list, label = "loss_G_GAN")
    plt.plot(loss_G_cycle_list, label = "loss_G_Cycle")
    plt.plot(loss_D_list, label = "loss_D_list")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    args = train_args()
    train(args)