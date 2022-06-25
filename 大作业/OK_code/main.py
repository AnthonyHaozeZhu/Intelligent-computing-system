# -*- coding: UTF-8 -*-
"""
@Project ：GAN 
@File ：main.py
@Author ：AnthonyZ
@Date ：2022/6/22 21:03
"""


import argparse
from tqdm import tqdm

from model import *
from data import *
from utils import *


def train(args):
    for epoch in range(args.epochs):
        dataloader = get_dataloader(args)
        train_tqdm = tqdm(dataloader, desc="Epoch {} / {}".format(epoch, args.epochs))

        discriminator.train()
        generator.train()

        for real_img, _ in train_tqdm:
            discriminator.zero_grad()
            mini_batch = real_img.shape[0]
            real_img = real_img.to(args.device)
            y_real_ = torch.ones(mini_batch).to(args.device)
            y_fake_ = torch.zeros(mini_batch).to(args.device)
            D_result = discriminator(real_img).squeeze()

            D_real_loss = BCE_loss(D_result, y_real_)
            z_ = torch.randn((mini_batch, 100)).to(args.device)
            G_result = generator(z_)

            D_result = discriminator(G_result).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            generator.zero_grad()

            z_ = torch.randn((mini_batch, 100)).to(args.device)

            G_result = generator(z_)
            D_result = discriminator(G_result).squeeze()
            G_train_loss = BCE_loss(D_result, y_real_)
            G_train_loss.backward()
            G_optimizer.step()

            train_tqdm.set_postfix({"lossD": "%.3g" % D_train_loss.item(), "lossG": "%.3g" % G_train_loss.item()})

        print("Validating...")
        generator.eval()
        print("Save example gen-image and model")
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), os.path.join(ckpt_dir, "generator-epoch{}.pt".format(epoch)))
            torch.save(discriminator.state_dict(), os.path.join(ckpt_dir, "discriminator-epoch{}.pt".format(epoch)))
        n_row = 10
        batches_done = epoch
        gen_imgs = generator(torch.randn(100, 100).to(args.device)).to("cpu")
        path = "image/" + str(batches_done) + ".png"
        save_image(gen_imgs.detach().numpy(), os.path.join(args.logdir, path), nrow=n_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="../Market1501/bounding_box_train", type=str, help="The input data dir")
    parser.add_argument("--logdir", default="./log", type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--device", default='cpu', type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=0.0002, type=float)
    parser.add_argument("--num_worker", default=10, type=float)
    args = parser.parse_args()

    generator = Generator(64).to(args.device)
    discriminator = Discriminator(64).to(args.device)
    generator.weight_init(mean=0.0, std=0.02)
    discriminator.weight_init(mean=0.0, std=0.02)
    BCE_loss = nn.BCELoss()

    G_optimizer = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    ckpt_dir = os.path.join(args.logdir, "checkpoint")
    image_dir = os.path.join(args.logdir, "image")

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    train(args)
