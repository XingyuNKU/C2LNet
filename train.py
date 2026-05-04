import copy
from PIL import Image
import numpy as np
from FusionNet import FusionNet
from TaskFusion_dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os.path as osp
import os
from logger import setup_logger
from loss import GeneratorLoss, DiscriminatorLoss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Dual_discriminator import DisIR_net, DisVIS_net
import matplotlib.pyplot as plt

def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

class FrozenBranch(nn.Module):
    def __init__(self, conv_module, drc1_module):
        super().__init__()
        self.conv = copy.deepcopy(conv_module)
        self.drc1 = copy.deepcopy(drc1_module)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.drc1(self.conv(x))

    def sync_from(self, conv_module, drc1_module):
        self.conv.load_state_dict(conv_module.state_dict())
        self.drc1.load_state_dict(drc1_module.state_dict())

def _plot_single_curve(x, y, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(
        x,
        y,
        color='blue',
        marker='o' if len(y) <= 50 else None,
        linewidth=1.5,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_loss_curves(save_dir, history, epoch_history):
    os.makedirs(save_dir, exist_ok=True)

    # ---------- iteration ----------
    _plot_single_curve(
        x=range(len(history['loss_gen'])),
        y=history['loss_gen'],
        title='Generator Adversarial Loss (Iteration)',
        xlabel='Iteration',
        ylabel='Loss',
        save_path=os.path.join(save_dir, 'iter_loss_gen.png'),
    )

    _plot_single_curve(
        x=range(len(history['loss_dis_ir'])),
        y=history['loss_dis_ir'],
        title='Discriminator IR Loss (Iteration)',
        xlabel='Iteration',
        ylabel='Loss',
        save_path=os.path.join(save_dir, 'iter_loss_dis_ir.png'),
    )

    _plot_single_curve(
        x=range(len(history['loss_dis_vis'])),
        y=history['loss_dis_vis'],
        title='Discriminator VIS Loss (Iteration)',
        xlabel='Iteration',
        ylabel='Loss',
        save_path=os.path.join(save_dir, 'iter_loss_dis_vis.png'),
    )

    # ---------- epoch  ----------
    _plot_single_curve(
        x=range(1, len(epoch_history['loss_gen']) + 1),
        y=epoch_history['loss_gen'],
        title='Generator Adversarial Loss (Epoch Avg)',
        xlabel='Epoch',
        ylabel='Average Loss',
        save_path=os.path.join(save_dir, 'epoch_loss_gen.png'),
    )

    _plot_single_curve(
        x=range(1, len(epoch_history['loss_dis_ir']) + 1),
        y=epoch_history['loss_dis_ir'],
        title='Discriminator IR Loss (Epoch Avg)',
        xlabel='Epoch',
        ylabel='Average Loss',
        save_path=os.path.join(save_dir, 'epoch_loss_dis_ir.png'),
    )

    _plot_single_curve(
        x=range(1, len(epoch_history['loss_dis_vis']) + 1),
        y=epoch_history['loss_dis_vis'],
        title='Discriminator VIS Loss (Epoch Avg)',
        xlabel='Epoch',
        ylabel='Average Loss',
        save_path=os.path.join(save_dir, 'epoch_loss_dis_vis.png'),
    )

    np.save(os.path.join(save_dir, 'loss_history.npy'), history, allow_pickle=True)
    np.save(os.path.join(save_dir, 'epoch_history.npy'), epoch_history, allow_pickle=True)

def train_fusion(logger=None, args=None):
    lr_start = 0.001
    modelpth = './model'
    Method = 'Fusion'
    modelpth = os.path.join(modelpth, Method)
    os.makedirs(modelpth, mode=0o777, exist_ok=True)

    fusionmodel = FusionNet(output=1).cuda()
    fusionmodel.train()

    generator_loss = GeneratorLoss()
    discriminator_loss = DiscriminatorLoss()

    frozen_vis_branch = FrozenBranch(fusionmodel.vis_conv, fusionmodel.vis_drc1).cuda()
    frozen_inf_branch = FrozenBranch(fusionmodel.inf_conv, fusionmodel.inf_drc1).cuda()
    frozen_vis_branch.eval()
    frozen_inf_branch.eval()

    dis_ir_model = DisIR_net(in_channels=32).cuda()
    dis_vis_model = DisVIS_net(in_channels=32).cuda()

    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    dis_ir_optimizer = torch.optim.Adam(
        dis_ir_model.parameters(), lr=lr_start * 0.1, betas=(0.5, 0.999)
    )
    dis_vis_optimizer = torch.optim.Adam(
        dis_vis_model.parameters(), lr=lr_start * 0.1, betas=(0.5, 0.999)
    )

    train_dataset = Fusion_dataset('train')
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    train_loader.n_iter = len(train_loader)

    epoch = 20
    st = glob_st = time.time()

    if logger is not None:
        logger.info('Training Fusion Model start~')

    history = {
        'loss_total': [],
        'loss_in': [],
        'loss_grad': [],
        'loss_gen': [],
        'loss_dis_ir': [],
        'loss_dis_vis': [],
        'loss_att': [],
        'loss_entropy': [],
    }
    epoch_history = {k: [] for k in history.keys()}

    for epo in range(0, epoch):
        frozen_vis_branch.sync_from(fusionmodel.vis_conv, fusionmodel.vis_drc1)
        frozen_inf_branch.sync_from(fusionmodel.inf_conv, fusionmodel.inf_drc1)
        frozen_vis_branch.eval()
        frozen_inf_branch.eval()

        lr_base = 0.001
        lr_decay = 0.75
        lr_this_epo = lr_base * (lr_decay ** epo)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo
        for param_group in dis_ir_optimizer.param_groups:
            param_group['lr'] = lr_this_epo * 0.5
        for param_group in dis_vis_optimizer.param_groups:
            param_group['lr'] = lr_this_epo * 0.5

        epoch_losses = {k: [] for k in history.keys()}

        for it, (image_vis, image_ir, label, name) in enumerate(train_loader):
            image_vis = image_vis.cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = image_ir.cuda()
            label = label.cuda()

            fusionmodel.eval()
            dis_vis_model.train()
            dis_ir_model.train()

            with torch.no_grad():
                logits, att_source, a1, a2, fv3, fi3 = fusionmodel(image_vis_ycrcb, image_ir)
                x_vis_p1_real = att_source[4]
                x_inf_p1_real = att_source[5]
                f_vis_fake = frozen_vis_branch(logits)
                f_ir_fake = frozen_inf_branch(logits)

            dis_ir_optimizer.zero_grad()
            fake_ir_output = dis_ir_model(f_ir_fake.detach())
            loss_dis_ir_fake = discriminator_loss(fake_ir_output, isReal=False)
            real_ir_output = dis_ir_model(x_inf_p1_real.detach())
            loss_dis_ir_real = discriminator_loss(real_ir_output, isReal=True)
            loss_dis_ir = loss_dis_ir_fake + loss_dis_ir_real
            loss_dis_ir.backward()
            dis_ir_optimizer.step()

            dis_vis_optimizer.zero_grad()
            fake_vis_output = dis_vis_model(f_vis_fake.detach())
            loss_dis_vis_fake = discriminator_loss(fake_vis_output, isReal=False)
            real_vis_output = dis_vis_model(x_vis_p1_real.detach())
            loss_dis_vis_real = discriminator_loss(real_vis_output, isReal=True)
            loss_dis_vis = loss_dis_vis_real + loss_dis_vis_fake
            loss_dis_vis.backward()
            dis_vis_optimizer.step()

            fusionmodel.train()
            dis_vis_model.eval()
            dis_ir_model.eval()

            logits, att_source, a1, a2, fv3, fi3 = fusionmodel(image_vis_ycrcb, image_ir)
            x_vis_p1 = att_source[4]
            x_inf_p1 = att_source[5]

            f_vis_fake = frozen_vis_branch(logits)
            f_ir_fake = frozen_inf_branch(logits)

            ir_fake_output = dis_ir_model(f_ir_fake)
            vis_fake_output = dis_vis_model(f_vis_fake)

            fusion_ycrcb = torch.cat(
                (logits, image_vis_ycrcb[:, 1:2, :, :], image_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)

            optimizer.zero_grad()

            loss_fusion, loss_in, loss_grad, loss_gen, loss_att, loss_entropy = generator_loss(
                image_vis_ycrcb,
                image_ir,
                label,
                logits,
                ir_fake_output,
                vis_fake_output,
                att_source,
                [x_vis_p1, x_inf_p1],
            )

            loss_total = loss_fusion
            loss_total.backward()
            optimizer.step()

            vals = {
                'loss_total': loss_total.item(),
                'loss_in': loss_in.item(),
                'loss_grad': loss_grad.item(),
                'loss_gen': loss_gen.item() if torch.is_tensor(loss_gen) else float(loss_gen),
                'loss_dis_ir': loss_dis_ir.item(),
                'loss_dis_vis': loss_dis_vis.item(),
                'loss_att': loss_att.item() if torch.is_tensor(loss_att) else float(loss_att),
                'loss_entropy': loss_entropy.item() if torch.is_tensor(loss_entropy) else float(loss_entropy),
            }
            for k, v in vals.items():
                history[k].append(v)
                epoch_losses[k].append(v)

            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * epo + it + 1
            eta = int((train_loader.n_iter * epoch - now_it) * (glob_t_intv / now_it))
            eta = str(datetime.timedelta(seconds=eta))

            if now_it % 10 == 0:
                msg = (
                    f'step: {now_it}/{train_loader.n_iter * epoch}, '
                    f'loss_total: {vals["loss_total"]:.4f}, '
                    f'loss_in: {vals["loss_in"]:.4f}, '
                    f'loss_grad: {vals["loss_grad"]:.4f}, '
                    f'loss_gen: {vals["loss_gen"]:.4f}, '
                    f'loss_dis_ir: {vals["loss_dis_ir"]:.4f}, '
                    f'loss_dis_vis: {vals["loss_dis_vis"]:.4f}, '
                    f'loss_att: {vals["loss_att"]:.4f}, '
                    f'loss_entropy: {vals["loss_entropy"]:.4f}, '
                    f'eta: {eta}, time: {t_intv:.4f}'
                )
                if logger is not None:
                    logger.info(msg)
                else:
                    print(msg)
                st = ed

        for k in epoch_history.keys():
            epoch_history[k].append(float(np.mean(epoch_losses[k])))

        epoch_msg = (
            f"Epoch [{epo + 1}/{epoch}] "
            f"avg_loss_total: {epoch_history['loss_total'][-1]:.4f}, "
            f"avg_loss_gen: {epoch_history['loss_gen'][-1]:.4f}, "
            f"avg_loss_dis_ir: {epoch_history['loss_dis_ir'][-1]:.4f}, "
            f"avg_loss_dis_vis: {epoch_history['loss_dis_vis'][-1]:.4f}, "
            f"lr: {lr_this_epo:.6f}"
        )
        if logger is not None:
            logger.info(epoch_msg)
        else:
            print(epoch_msg)

    fusion_model_file = os.path.join(modelpth, 'fusion_model.pth')
    torch.save(fusionmodel.state_dict(), fusion_model_file)
    if logger is not None:
        logger.info("Fusion Model Save to: {}".format(fusion_model_file))

    curve_dir = os.path.join('./logs', 'curves')
    save_loss_curves(curve_dir, history, epoch_history)
    if logger is not None:
        logger.info("Loss curves saved to: {}".format(curve_dir))
        logger.info('\n')

def run_fusion(type='train', args=None):
    fusion_model_path = './model/Fusion/fusion_model.pth'
    fused_dir = os.path.join('./MSRS/Fusion', type, 'MSRS')
    os.makedirs(fused_dir, mode=0o777, exist_ok=True)

    fusionmodel = FusionNet(output=1)
    fusionmodel.eval()

    if args.gpu >= 0:
        fusionmodel.cuda(args.gpu)
        map_location = f'cuda:{args.gpu}'
    else:
        map_location = 'cpu'

    fusionmodel.load_state_dict(torch.load(fusion_model_path, map_location=map_location))
    print('Fusion model loaded!')

    test_dataset = Fusion_dataset(type)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)

    with torch.no_grad():
        for it, (images_vis, images_ir, labels, name) in enumerate(test_loader):
            images_vis = images_vis.cuda(args.gpu) if args.gpu >= 0 else images_vis
            images_ir = images_ir.cuda(args.gpu) if args.gpu >= 0 else images_ir
            labels = labels.cuda(args.gpu) if args.gpu >= 0 else labels

            images_vis_ycrcb = RGB2YCrCb(images_vis)
            logits, _, _, _, _, _ = fusionmodel(images_vis_ycrcb, images_ir)

            fusion_ycrcb = torch.cat(
                (logits, images_vis_ycrcb[:, 1:2, :, :], images_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)

            ones = torch.ones_like(fusion_image)
            zeros = torch.zeros_like(fusion_image)
            fusion_image = torch.where(fusion_image > ones, ones, fusion_image)
            fusion_image = torch.where(fusion_image < zeros, zeros, fusion_image)

            fused_image = fusion_image.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                np.max(fused_image) - np.min(fused_image) + 1e-8
            )
            fused_image = np.uint8(255.0 * fused_image)

            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = image.squeeze()
                image = Image.fromarray(image)
                save_path = os.path.join(fused_dir, name[k])
                image.save(save_path)
                print('Fusion {0} Successfully!'.format(save_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='SD3Fuse')
    parser.add_argument('--batch_size', '-B', type=int, default=10)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()

    logpath = './logs'
    os.makedirs(logpath, mode=0o777, exist_ok=True)
    logger = logging.getLogger()
    setup_logger(logpath)

    train_fusion(logger, args)
    print("Train Fusion Model Successfully~!")

    run_fusion('train', args)
    print("Fusion Image Successfully~!")

    print("Training Done!")