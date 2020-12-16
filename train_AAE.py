

import torch.utils.data
from defaults import get_cfg_defaults
from torch import optim
from torchvision.utils import save_image
from torch.autograd import Variable
import time
import logging
import os
from dataloading import make_datasets, make_dataloader
from net import Generator, Discriminator, Encoder, ZDiscriminator_mergebatch, ZDiscriminator,LatentZ
from utils.tracker import LossTracker
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def train(folding_id, inliner_classes, ic):
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/mnist.yaml')
    cfg.freeze()
    logger = logging.getLogger("logger")

    zsize = cfg.MODEL.LATENT_SIZE
    #print("zsize: "+str(zsize))
    output_folder = os.path.join('results_' + str(folding_id) + "_" + "_".join([str(x) for x in inliner_classes]))
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    train_set, _, _ = make_datasets(cfg, folding_id, inliner_classes)

    logger.info("Train set size: %d" % len(train_set))

    G = Generator(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)
    G.weight_init(mean=0, std=0.02)

    D = Discriminator(channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)
    D.weight_init(mean=0, std=0.02)

    #LZ=LatentZ(1,1)

    E = Encoder(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)
    E.weight_init(mean=0, std=0.02)

    if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH:
        ZD = ZDiscriminator_mergebatch(zsize, cfg.TRAIN.BATCH_SIZE)
    else:
        ZD = ZDiscriminator(zsize, cfg.TRAIN.BATCH_SIZE)
    ZD.weight_init(mean=0, std=0.02)

    lr = cfg.TRAIN.BASE_LEARNING_RATE
    #lr=0.0001


    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    GE_optimizer = optim.Adam(list(E.parameters()) + list(G.parameters()), lr=lr, betas=(0.5, 0.999))
    ZD_optimizer = optim.Adam(ZD.parameters(), lr=lr, betas=(0.5, 0.999))

    BCE_loss = nn.BCELoss()
    sample = torch.randn(64, zsize).view(-1, zsize, 1, 1)

    tracker = LossTracker(output_folder=output_folder)

    #cfg.TRAIN.EPOCH_COUNT
    for epoch in range(100):
        G.train()
        D.train()
        E.train()
        ZD.train()

        epoch_start_time = time.time()
        print(" cfg.TRAIN.BATCH_SIZE :"+str( cfg.TRAIN.BATCH_SIZE))
        data_loader = make_dataloader(train_set, cfg.TRAIN.BATCH_SIZE, torch.cuda.current_device())
        train_set.shuffle()

        #if (epoch + 1) % 30 == 0:
            #G_optimizer.param_groups[0]['lr'] /= 4
            #D_optimizer.param_groups[0]['lr'] /= 4
            #GE_optimizer.param_groups[0]['lr'] /= 4
            #ZD_optimizer.param_groups[0]['lr'] /= 4
            #print("learning rate change!")

        for y, x in data_loader:
            x = x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE)
            
            #print("x: "+str(x.size()))

            y_real_ = torch.ones(x.shape[0])
            y_fake_ = torch.zeros(x.shape[0])
            
            #print(" y r: "+str(y_real_.size()))
            #print("y f: "+str(y_fake_.size()))

            y_real_z = torch.ones(1 if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH else x.shape[0])
            y_fake_z = torch.zeros(1 if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH else x.shape[0])
            
            #print("y real z: "+str(y_real_z.size()))
            #print("y fake z: "+str(y_fake_z.size()))

            #############################################

            D.zero_grad()

            D_result = D(x).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)

            z = torch.randn((x.shape[0], zsize)).view(-1, zsize, 1, 1)
            z = Variable(z)

            x_fake = G(z).detach()
            D_result = D(x_fake).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)

            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()

            D_optimizer.step()

            tracker.update(dict(D=D_train_loss))


            #############################################

            G.zero_grad()

            z = torch.randn((x.shape[0], zsize)).view(-1, zsize, 1, 1)
            z = Variable(z)

            x_fake = G(z)
            D_result = D(x_fake).squeeze()

            G_train_loss = BCE_loss(D_result, y_real_)

            G_train_loss.backward()
            G_optimizer.step()

            tracker.update(dict(G=G_train_loss))

            #############################################

            ZD.zero_grad()

            z = torch.randn((x.shape[0], zsize)).view(-1, zsize)
            z = Variable(z)

            ZD_result = ZD(z).squeeze()
            ZD_real_loss = BCE_loss(ZD_result, y_real_z)

            z,logvar,mu = E(x)
            z=z.squeeze().detach()

            ZD_result = ZD(z).squeeze()
            ZD_fake_loss = BCE_loss(ZD_result, y_fake_z)

            ZD_train_loss = ZD_real_loss + ZD_fake_loss
            ZD_train_loss.backward()

            ZD_optimizer.step()

            tracker.update(dict(ZD=ZD_train_loss))

            # #############################################

            E.zero_grad()
            G.zero_grad()

            #z = E(x)
            #x_d = G(z)

            z,logvar,mu = E(x)
            x_d = G(z)
            
            #print("------------------------ here it goes")
            #print(str(x.size()))
            #print(str(p_x.size()))
            #print(str(z.size()))
            #print(str(x_d.size()))
            #print("------------------------ here it goes")

            ZD_result = ZD(z.squeeze()).squeeze()

            E_train_loss = BCE_loss(ZD_result, y_real_z) * 1.0

            Recon_loss = F.binary_cross_entropy(x_d, x.detach(),reduction="sum")*2.0
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            GE_loss=Recon_loss+kl_loss

            (GE_loss+E_train_loss).backward()

            GE_optimizer.step()

            tracker.update(dict(GE=GE_loss,RE=Recon_loss,K=kl_loss, E=E_train_loss))

            # #############################################

        comparison = torch.cat([x, x_d])
        save_image(comparison.cpu(), os.path.join(output_folder, 'reconstruction_' + str(epoch) + '.png'), nrow=x.shape[0])

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        logger.info('[%d/%d] - ptime: %.2f, %s' % ((epoch + 1), 100, per_epoch_ptime, tracker))

        tracker.register_means(epoch)
        tracker.plot()

        with torch.no_grad():
            resultsample = G(sample).cpu()
            save_image(resultsample.view(64,
                                         cfg.MODEL.INPUT_IMAGE_CHANNELS,
                                         cfg.MODEL.INPUT_IMAGE_SIZE,
                                         cfg.MODEL.INPUT_IMAGE_SIZE),
                       os.path.join(output_folder, 'sample_' + str(epoch) + '.png'))

    logger.info("Training finish!... save training results")

    os.makedirs("models", exist_ok=True)

    print("Training finish!... save training results")
    torch.save(G.state_dict(), "models/Gmodel_%d_%d.pkl" %(folding_id, ic))
    torch.save(E.state_dict(), "models/Emodel_%d_%d.pkl" %(folding_id, ic))
    #torch.save(D.state_dict(), "Dmodel_%d_%d.pkl" %(folding_id, ic))
    #torch.save(ZD.state_dict(), "ZDmodel_%d_%d.pkl" %(folding_id, ic))


