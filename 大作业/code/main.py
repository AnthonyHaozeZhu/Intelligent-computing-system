# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File ：main.py.py
@Author ：AnthonyZ
@Date ：2022/6/18 19:59
"""

import argparse
from torch.autograd import Variable

from utils import *
from data import *
from model import *

from tqdm import tqdm
import os


def train(args):
    for i in range(args.epochs):
        train_tqdm = tqdm(gan_loader, desc="Epoch " + str(i))
        for index, real_imgs in enumerate(train_tqdm):
            valid = Variable(torch.FloatTensor(real_imgs.shape[0], 1).fill_(1.0), requires_grad=False).to(args.device)
            fake = Variable(torch.FloatTensor(real_imgs.shape[0], 1).fill_(0.0), requires_grad=False).to(args.device)

            optimizer_G.zero_grad()

            z = Variable(torch.FloatTensor(np.random.normal(0, 1, (real_imgs.shape[0], 100)))).to(args.device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            real_dis = discriminator(real_imgs.to(args.device))
            fake_dis = discriminator(gen_imgs.detach())
            real_loss = adversarial_loss(real_dis, valid)
            fake_loss = adversarial_loss(fake_dis, fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            train_tqdm.set_postfix({"lossD": "%.3g" % d_loss.detch().item(), "lossG": "%.3g" % g_loss.detch().item()})

        # if i % 10 == 0:
        print("Save example gen-image and model")
        torch.save(generator.state_dict(), os.path.join(args.logdir, "saved_models/generator_last.pt"))
        torch.save(discriminator.state_dict(), os.path.join(args.logdir, "saved_models/discriminator_last.pt"))
        n_row = 10
        batches_done = i
        gen_imgs = generator(torch.randn(100, 100).to(args.device)).to("cpu")
        path = "example/" + str(batches_done) + ".png"
        save_image(gen_imgs.detach().numpy(), os.path.join(args.logdir, path), nrow=n_row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--Gan_data_path", default="./Market1501/pre_load", type=str, help="The input data dir")
    parser.add_argument("--logdir", default="./log", type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--device", default='cpu', type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=0.0002, type=float)
    parser.add_argument("--channels", default=3, type=int, help="The number of channels of the image")
    parser.add_argument("--img_w", default=256, type=int)
    parser.add_argument("--img_h", default=128, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
        os.makedirs(os.path.join(args.logdir, "saved_models"))
        os.makedirs(os.path.join(args.logdir, "example"))

    gan_loader = DataLoader(GanLoader(args), batch_size=args.batch_size, shuffle=True)

    generator = Generator(args).to(args.device)
    discriminator = Discriminator(args).to(args.device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    # criterion = nn.MSELoss()
    adversarial_loss = nn.MSELoss().to(args.device)

    train(args)

