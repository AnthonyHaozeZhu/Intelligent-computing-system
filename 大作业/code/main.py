# import argparse
# from torch.autograd import Variable
#
# from utils import *
# from data import *
# from model import *
#
# from tqdm import tqdm
# import os
#
#
# def train(args):
#     for i in range(args.epochs):
#         train_tqdm = tqdm(gan_loader, desc="Epoch " + str(i))
#         for index, real_imgs in enumerate(train_tqdm):
#             valid = Variable(torch.FloatTensor(real_imgs.shape[0], 1).fill_(1.0), requires_grad=False).to(args.device)
#             fake = Variable(torch.FloatTensor(real_imgs.shape[0], 1).fill_(0.0), requires_grad=False).to(args.device)
#
#             optimizer_G.zero_grad()
#
#             z = Variable(torch.FloatTensor(np.random.normal(0, 1, (real_imgs.shape[0], 100)))).to(args.device)
#             gen_imgs = generator(z)
#             g_loss = adversarial_loss(discriminator(gen_imgs), valid)
#
#             g_loss.backward()
#             optimizer_G.step()
#
#             optimizer_D.zero_grad()
#
#             real_dis = discriminator(real_imgs.to(args.device))
#             fake_dis = discriminator(gen_imgs.detach())
#             real_loss = adversarial_loss(real_dis, valid)
#             fake_loss = adversarial_loss(fake_dis, fake)
#             d_loss = (real_loss + fake_loss) / 2
#
#             d_loss.backward()
#             optimizer_D.step()
#
#             train_tqdm.set_postfix({"lossD": "%.3g" % d_loss.item(), "lossG": "%.3g" % g_loss.item()})
#
#         # if i % 10 == 0:
#         print("Save example gen-image and model")
#         torch.save(generator.state_dict(), os.path.join(args.logdir, "saved_models/generator_last.pt"))
#         torch.save(discriminator.state_dict(), os.path.join(args.logdir, "saved_models/discriminator_last.pt"))
#         n_row = 10
#         batches_done = i
#         gen_imgs = generator(torch.randn(100, 100).to(args.device)).to("cpu")
#         path = "example/" + str(batches_done) + ".png"
#         save_image(gen_imgs.detach().numpy(), os.path.join(args.logdir, path), nrow=n_row)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--Gan_data_path", default="../Market1501/pre_load", type=str, help="The input data dir")
#     parser.add_argument("--logdir", default="./log", type=str)
#     parser.add_argument("--epochs", default=100, type=int)
#     parser.add_argument("--device", default='mps', type=str)
#     parser.add_argument("--batch_size", default=128, type=int)
#     parser.add_argument("--learning_rate", default=0.0002, type=float)
#     parser.add_argument("--channels", default=3, type=int, help="The number of channels of the image")
#     parser.add_argument("--img_w", default=128, type=int)
#     parser.add_argument("--img_h", default=128, type=int)
#
#     args = parser.parse_args()
#
#     if not os.path.exists(args.logdir):
#         os.makedirs(args.logdir)
#         os.makedirs(os.path.join(args.logdir, "saved_models"))
#         os.makedirs(os.path.join(args.logdir, "example"))
#
#     gan_loader = torch.utils.data.DataLoader(GanLoader(args), batch_size=args.batch_size, shuffle=True)
#
#     generator = Generator().to(args.device)
#     discriminator = Discriminator().to(args.device)
#
#     optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
#     optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
#
#     # criterion = nn.MSELoss()
#     adversarial_loss = nn.MSELoss().to(args.device)
#
#     train(args)



import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from utils import *
from data import *
from model import *

from tqdm import tqdm
import os


def train(opt):
    dataloader = get_dataloader(opt)
    generator = Generator(1, 3)
    discriminator = Discriminator(3)
    generator.to(opt.device)
    discriminator.to(opt.device)
    step = 0
    ckpt_dir = os.path.join(opt.logdir, "checkpoint")
    image_dir = os.path.join(opt.logdir, "image")

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    writer = SummaryWriter(os.path.join(opt.logdir, "tensorboard"))
    optimizer_G = torch.optim.SGD(generator.parameters(), lr=opt.learning_rate)
    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=opt.learning_rate)
    loss_f = nn.BCELoss()
    for epoch in range(opt.epoch):
        discriminator.train()
        generator.train()
        train_tqdm = tqdm(dataloader, desc="Epoch {} / {}".format(epoch, opt.epoch))
        for image, _ in train_tqdm:
            image = image.to(opt.device)
            # vec = vec.to(opt.device)
            vec = torch.normal(0, 1, (image.shape[0], 100)).to(opt.device)

            # D
            # for _ in range(10):
            generator.eval()
            discriminator.train()
            optimizer_D.zero_grad()
            # optimizer_G.zero_grad()
            gen_imgs_w = generator(vec)
            discr_res_w = discriminator(gen_imgs_w)
            dis_loss_w = loss_f(discr_res_w, torch.zeros_like(discr_res_w))
            discr_res_c = discriminator(image)
            dis_loss_c = loss_f(discr_res_c, torch.ones_like(discr_res_c))
            d_loss = (dis_loss_w + dis_loss_c) / 2
            d_loss.backward()
            optimizer_D.step()

            generator.train()
            discriminator.eval()
            # G
            optimizer_G.zero_grad()
            # optimizer_D.zero_grad()
            # vec = torch.normal(0, 1, (image.shape[0], 100)).to(opt.device)
            gen_imgs = generator(vec)
            discr_res = discriminator(gen_imgs)
            g_loss = loss_f(discr_res, torch.ones_like(discr_res))
            g_loss.backward()
            optimizer_G.step()

            train_tqdm.set_postfix({"lossD": "%.3g" % d_loss.item(), "lossG": "%.3g" % g_loss.item()})
            writer.add_scalar("LossG", g_loss.item(), step)
            writer.add_scalar("LossD", d_loss.item(), step)
            step += 1

        if epoch % 10 == 0:
            print("Save example gen-image and model")
            torch.save(generator.state_dict(), os.path.join(ckpt_dir, "G-epoch{}.pkl".format(epoch)))
            torch.save(discriminator.state_dict(), os.path.join(ckpt_dir, "D-epoch{}.pkl".format(epoch)))

        print("Validating...")
        generator.eval()
        print("Save example gen-image and model")
        torch.save(generator.state_dict(), os.path.join(args.logdir, "saved_models/generator_last.pt"))
        torch.save(discriminator.state_dict(), os.path.join(args.logdir, "saved_models/discriminator_last.pt"))
        n_row = 10
        batches_done = epoch
        gen_imgs = generator(torch.randn(100, 100).to(args.device)).to("cpu")
        path = "example/" + str(batches_done) + ".png"
        save_image(gen_imgs.detach().numpy(), os.path.join(args.logdir, path), nrow=n_row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="../Market1501/bounding_box_train", type=str, help="The input data dir")
    parser.add_argument("--logdir", default="./log", type=str)
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--device", default='mps', type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--num_worker", default=10, type=float)
    # parser.add_argument("--channels", default=3, type=int, help="The number of channels of the image")
    # parser.add_argument("--img_w", default=128, type=int)
    # parser.add_argument("--img_h", default=128, type=int)

    args = parser.parse_args()
    train(args)