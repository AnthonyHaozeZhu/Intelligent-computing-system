# -*- coding: UTF-8 -*-
"""
@Project ：GAN 
@File ：main.py
@Author ：AnthonyZ
@Date ：2022/6/22 21:03
"""


import argparse
from tqdm import tqdm

from torchvision import datasets
from torchvision import transforms as tsf

from model import *
from data import *
from utils import *


def train(args):
    for epoch in range(args.epochs):
        dataloader = get_dataloader(args)
        train_tqdm = tqdm(dataloader, desc="Epoch {} / {}".format(epoch, args.epochs))
        D.train()
        G.train()
        for x_, _ in train_tqdm:
            D.zero_grad()
            mini_batch = x_.shape[0]
            x_ = x_.to(args.device)
            y_real_ = torch.ones(mini_batch).to(args.device)
            y_fake_ = torch.zeros(mini_batch).to(args.device)
            D_result = D(x_).squeeze()

            # print(D_result.shape)

            D_real_loss = BCE_loss(D_result, y_real_)
            z_ = torch.randn((mini_batch, 100)).to(args.device)
            G_result = G(z_)

            D_result = D(G_result).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)
            D_fake_score = D_result.data.mean()

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            G.zero_grad()

            z_ = torch.randn((mini_batch, 100)).to(args.device)

            G_result = G(z_)
            D_result = D(G_result).squeeze()
            G_train_loss = BCE_loss(D_result, y_real_)
            G_train_loss.backward()
            G_optimizer.step()

            # lr1.step()
            # lr2.step()

            train_tqdm.set_postfix({"lossD": "%.3g" % D_train_loss.item(), "lossG": "%.3g" % G_train_loss.item()})

        if (epoch+1) % 10 == 0:
            print("Save example gen-image and model")
            torch.save(G.state_dict(), os.path.join(ckpt_dir, "G-epoch{}.pkl".format(epoch)))
            torch.save(D.state_dict(), os.path.join(ckpt_dir, "D-epoch{}.pkl".format(epoch)))

        print("Validating...")
        G.eval()
        print("Save example gen-image and model")
        # torch.save(G.state_dict(), os.path.join(args.logdir, "checkpoint/generator_last.pt"))
        # torch.save(D.state_dict(), os.path.join(args.logdir, "checkpoint/discriminator_last.pt"))
        n_row = 10
        batches_done = epoch
        gen_imgs = G(torch.randn(100, 100).to(args.device)).to("cpu")
        path = "image/" + str(batches_done) + ".png"
        save_image(gen_imgs.detach().numpy(), os.path.join(args.logdir, path), nrow=n_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="./Market1501/bounding_box_train", type=str, help="The input data dir")
    parser.add_argument("--logdir", default="./log", type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--device", default='cpu', type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=0.0002, type=float)
    parser.add_argument("--num_worker", default=10, type=float)
    args = parser.parse_args()

    G = Generator(64).to(args.device)
    D = Discriminator(64).to(args.device)
    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
    BCE_loss = nn.BCELoss()

    G_optimizer = torch.optim.Adam(G.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    D_optimizer = torch.optim.Adam(D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    ckpt_dir = os.path.join(args.logdir, "checkpoint")
    image_dir = os.path.join(args.logdir, "image")

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    train(args)
