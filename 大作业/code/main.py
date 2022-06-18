# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File ：main.py.py
@Author ：AnthonyZ
@Date ：2022/6/18 19:59
"""

import argparse

from utils import *
from data import *
from model import *

from tqdm import tqdm
import os


def train(args):
    for i in range(args.epochs):
        train_tqdm = tqdm(gan_loader, desc="Epoch " + str(i))
        for index, real_imgs in enumerate(train_tqdm):
            real_imgs = real_imgs.to(args.device)
            optimizer_G.zero_grad()
            noise = torch.randn((real_imgs.shape[0], 100)).to(args.device)
            label_fake = torch.ones((real_imgs.shape[0], 1)).to(args.device)
            label_real = torch.zeros((real_imgs.shape[0], 1)).to(args.device)

            fake_imgs = generator(noise).detach()

            predict = discriminator(fake_imgs)
            lossG = criterion(predict, label_fake)
            lossG.backward()

            optimizer_G.step()

            # 判别器部分
            optimizer_D.zero_grad()
            predict_real = discriminator(real_imgs)
            lossD_real = criterion(predict_real, label_real)
            predict_fake = discriminator(fake_imgs)
            lossD_fake = criterion(predict_fake, label_fake)
            lossD = (lossD_fake + lossD_real) / 2
            lossD.backward()
            train_tqdm.set_postfix({"lossD": "%.3g" % lossD.item(), "lossG": "%.3g" % lossG.item()})

            optimizer_D.step()

        # if i % 10 == 0:
        print("Save example gen-image and model")
        torch.save(generator.state_dict(), os.path.join(args.logdir, "saved_models/generator_last.pt"))
        torch.save(discriminator.state_dict(), os.path.join(args.logdir, "saved_models/discriminator_last.pt"))
        n_row = 10
        batches_done = i
        # labels_temp = jittor.array(np.array([num for _ in range(n_row) for num in range(n_row)])).float32().stop_grad()
        # gen_imgs = generator(jittor.array(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))).float32().stop_grad())
        gen_imgs = generator(torch.randn(100, 100).to(args.device)).to("cpu")
        path = "example/%d.png" + str(batches_done) + ".png"
        save_image(gen_imgs.numpy(), os.path.join(args.logdir, path), nrow=n_row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--Gan_data_path", default="./Market1501/bounding_box_train", type=str, help="The input data dir")
    parser.add_argument("--logdir", default="./log", type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--device", default='cpu', type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
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

    criterion = nn.MSELoss()

    train(args)

