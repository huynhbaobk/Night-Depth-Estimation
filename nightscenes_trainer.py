# Author: Akhil Gurram
# Build on top of the monodepth2
# (Automatically pulled from git repo, monodepth2 source code is not included in this repository)
# This is the training script of the MonoDEVSNet framework.
# MonoDEVSNet: Monocular Depth Estimation through Virtual-world Supervision and Real-world SfM Self-Supervision
# https://arxiv.org/abs/2103.12209

# MIT License
#
# Copyright (c) 2021 Huawei Technologies Duesseldorf GmbH
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
example command line arguments:
 python3 monodevnset_trainer.py --cuda_idx 0 --num_workers 0 --batch_size 10 --height 192 --width 640 --max_depth 80 \
 --use_dc --use_le --use_ms --version markX --num_epochs 200 --png \
 --real_dataset kitti --syn_dataset vk_2.0 --real_data_path /mnt/largedisk/Datasets/KITTI \
 --syn_data_path /mnt/largedisk/Datasets
"""

import json
import os
import shutil
import sys
import time
from copy import deepcopy
from os.path import expanduser

import cv2
import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from PIL import Image

import networks
import monodepth2
from monodepth2 import KITTIRAWDataset, KITTIDepthDataset, KITTIOdomDataset
from monodepth2.evaluate_depth import compute_errors
from monodepth2.layers import disp_to_depth, compute_depth_errors
from monodepth2.trainer import Trainer
from monodepth2.utils import normalize_image, readlines, sec_to_hm_str
from utils import get_n_params
from utils.monodevsnet_options import MonoDEVSOptions
from utils import freeze_model, unfreeze_model, ImagePool


class MonoDEVSNetTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MonoDEVSNetTrainer, self).__init__(*args, **kwargs)

        # Load experiments options/parameters
        self.opt.trainer_name = 'MonoDEVSNetTrainer'
        self.opt.use_pose_net = self.use_pose_net

        # Set cuda index
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.opt.cuda_idx)
        self.device = torch.device("cuda:" + str(self.opt.cuda_idx) if torch.cuda.is_available() else "cpu")

        # Remove unnecessary variables from self object
        for attr in ('models', 'model_optimizer', 'parameters_to_train', 'model_lr_scheduler',
                     'dataset', 'train_loader', 'val_loader', 'val_iter'):
            self.__dict__.pop(attr, None)
        self.models, self.model_optimizer, self.parameters_to_train = {}, {}, []

        # Load network architecture
        ''' Setting Models, initialization and optimization '''
        self.torch_zero = torch.tensor(0.).float().to(self.device)
        self.True_ = torch.tensor(np.ones(self.opt.batch_size)).float().to(self.device)
        self.False_ = torch.tensor(np.zeros(self.opt.batch_size)).float().to(self.device)

        # Encoder for (depth, segmentation, pose)
        self.models["encoder"] = self.network_selection('encoder')
        self.parameters_to_train += list(self.models["encoder"].parameters())

        # Depth decoder
        self.models["depth_decoder"] = self.network_selection('depth_decoder')
        self.parameters_to_train += list(self.models["depth_decoder"].parameters())

        # Pose Encoder and Decoder
        if self.opt.use_pose_net:
            self.models["pose_encoder_real"] = self.network_selection('pose_encoder')
            self.parameters_to_train += list(self.models["pose_encoder_real"].parameters())
            self.models["pose_real"] = self.network_selection('pose')
            self.parameters_to_train += list(self.models["pose_real"].parameters())

            self.models["pose_encoder_syn"] = self.network_selection('pose_encoder')
            self.parameters_to_train += list(self.models["pose_encoder_syn"].parameters())
            self.models["pose_syn"] = self.network_selection('pose')
            self.parameters_to_train += list(self.models["pose_syn"].parameters())

        # Domain classifier
        self.lambda_ = 1
        if self.opt.use_dc:
            self.models["domain_classifier"] = self.network_selection('domain_classifier')
            self.parameters_to_train += list(self.models["domain_classifier"].parameters())

        # Set optimization parameters
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        # self.model_lr_scheduler =  optim.lr_scheduler.CosineAnnealingLR(self.model_optimizer, T_max=4, eta_min=1e-5)

        self.models["depth_classifier"] = self.network_selection('depth_classifier')
        self.depth_classifier_parameters = self.models["depth_classifier"].parameters()
        self.depth_classifier_optim = optim.Adam(self.depth_classifier_parameters, lr=self.opt.learning_rate)
        # self.image_pool = ImagePool(50)

        # register image coordinates
        if self.opt.use_position_map:
            h, w = self.opt.height, self.opt.width
            self.height_map = torch.arange(h).view(1, 1, h, 1).repeat(1, 1, 1, w) / (h - 1)
            self.width_map = torch.arange(w).view(1, 1, 1, w).repeat(1, 1, h, 1) / (w - 1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("[INFO] Train model on device : {} \n  ", self.device)
        print("[INFO] number of training parameters for each model")
        for model_name, model in self.models.items():
            # print('{:^15}: {:^15}: {:>5.2f} M'.format(model_name, self.opt.models_fcn_name[model_name],
            #                                           get_n_params(model) / 1000000))
            print('{:^15}: {:>5.2f} M'.format(model_name,
                            get_n_params(model) / 1000000))

        #################################### DATASETS ####################################

        # Load dataloader
        # Setting Datasets
        img_ext = '.png' if self.opt.png else '.jpg'

        self.syn_or_real = ''
        datasets_dict = {"kitti": KITTIRAWDataset,
                         "kitti_odom": KITTIOdomDataset,
                         "kitti_depth": KITTIDepthDataset}

        self.real_dataset = datasets_dict[self.opt.real_dataset]
        self.syn_dataset = datasets_dict[self.opt.syn_dataset]

        ### Training dataset ###
        ### NIGHT
        real_f_path = os.path.join(os.path.dirname(__file__), "monodepth2/splits", self.opt.real_split, "{}_files.txt")
        real_filenames = readlines(real_f_path.format("train_all"))
        real_train_dataset = self.real_dataset(data_path=self.opt.real_data_path, filenames=real_filenames,
                                               height=self.opt.height, width=self.opt.width,
                                               frame_idxs=self.opt.frame_ids, num_scales=4, is_train=True,
                                               img_ext=img_ext)

        
        ### DAY
        syn_f_path =  os.path.join(os.path.dirname(__file__), "monodepth2/splits", self.opt.syn_split, "{}_files.txt")
        syn_filenames = readlines(syn_f_path.format("train_all"))
        syn_train_dataset = self.syn_dataset(data_path=self.opt.syn_data_path, filenames=syn_filenames,
                                               height=self.opt.height, width=self.opt.width,
                                               frame_idxs=self.opt.frame_ids, num_scales=4, is_train=True,
                                               img_ext=img_ext)

        #### Validation dataset ###
        # NIGHT
        real_filenames = readlines(real_f_path.format("val_night"))
        real_val_dataset = self.real_dataset(data_path=self.opt.real_data_path, filenames=real_filenames,
                                             height=self.opt.height, width=self.opt.width,
                                             frame_idxs=[0], num_scales=4, is_train=False,
                                             img_ext=img_ext)

        # DAY
        syn_filenames = readlines(syn_f_path.format("val_day"))
        syn_val_dataset = self.syn_dataset(data_path=self.opt.syn_data_path, filenames=syn_filenames,
                                             height=self.opt.height, width=self.opt.width,
                                             frame_idxs=[0], num_scales=4, is_train=False,
                                             img_ext=img_ext)

        ### Training loader ###
        # NIGHT
        self.real_train_loader = DataLoader(
            real_train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        # DAY
        self.syn_train_loader = DataLoader(
            syn_train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        ### Validation loader ###
        # NIGHT 
        self.real_val_loader = DataLoader(
            real_val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)
        
        # DAY
        self.syn_val_loader = DataLoader(
            syn_val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)


        ### Training iteration approach ###
        self.real_train_iter, self.syn_train_iter = iter(self.real_train_loader), iter(self.syn_train_loader)
        
        ### val iteration approach ###
        self.real_val_iter, self.syn_val_iter = iter(self.real_val_loader), iter(self.syn_val_loader)

        self.num_total_files = min(real_train_dataset.__len__(), syn_train_dataset.__len__())
        self.num_total_steps = self.num_total_files // self.opt.batch_size * self.opt.num_epochs
        self.num_total_batch = self.num_total_files // self.opt.batch_size

        self.current_lr = self.opt.learning_rate

        #################################### Setting tensorboard ####################################
        self.opt.model_name = (self.opt.models_fcn_name['encoder'] + str(self.opt.num_layers) +
                               '_dc' + str(self.opt.use_dc)[0] +
                               '_le' + str(self.opt.use_le)[0] +
                               '_' + str(self.opt.width) + 'x' + str(self.opt.height) +
                               '_' + self.opt.version).replace(' ', '')
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        print('log directory path: {}'.format(self.log_path))

        # Setting tensorboard
        self.writers = {}
        for mode in ["train", "val_real", "val_syn"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        # Additional loss functions 
        self.L1Loss = nn.L1Loss().to(self.device)
        self.L2Loss = nn.MSELoss().to(self.device)
        self.CrossEntropy = nn.CrossEntropyLoss().to(self.device)
        self.LossDomainClassifier = nn.NLLLoss().to(self.device)
        self.gan_loss = networks.GANLoss('lsgan').to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("[INFO] Using night split:\n  ", self.opt.real_split)
        print("[INFO] Using day training split:\n  ", self.opt.syn_split)
        print("[INFO] There are night: {:d}, day: {:d} training items and "
              "night: {:d}, day: {:d} validation items\n".
              format(len(real_train_dataset), len(syn_train_dataset),
                     len(real_val_dataset), len(syn_val_dataset)))

        # train def
        self.step, self.epoch, self.previous_sp_loss = 0, 0, 0
        self.early_phase, self.mid_phase, self.late_phase = False, False, False
        self.start_time = time.time()
        # print(self.opt)
        self.save_opts()

    def train(self):
        # Run entire pipeline
        try:
            for self.epoch in range(self.opt.num_epochs):
                # Zeros optimization on all model_optimizer
                print("[INFO] - Run epoch " + str(self.epoch))
                self.zero_grad()
                self.run_epoch()

                if self.epoch >= 15:
                    self.opt.save_frequency = 1

                if self.epoch % self.opt.save_frequency == 0:
                    self.save_model()
                    
            # self.save_model()
            # save_best_model(self)

        except (KeyboardInterrupt, SystemExit) as e:
            print(e)
            sys.exit(0)


    def zero_grad(self):
        self.model_optimizer.zero_grad()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print(f"**********Training epoch {self.epoch}**********")
        self.set_train()

        for batch_idx in range(0, self.num_total_batch):

            before_op_time = time.time()
            # Choosing the dataloader for training model
            if self.choosing_dataset_to_train_with(batch_idx):
                # Synthetic dataset
                self.syn_or_real = 'syn'
                try:
                    inputs = self.syn_train_iter.__next__()
                except StopIteration:
                    print('Stopped as the iteration has reached to the END, and reloading the synthetic dataloader')
                    self.syn_train_iter = iter(self.syn_train_loader)
                    inputs = self.syn_train_iter.__next__()
            else:
                # Real dataset
                self.syn_or_real = 'real'
                try:
                    inputs = self.real_train_iter.__next__()
                except StopIteration:
                    print('Stopped as the iteration has reached to the END, and reloading the real dataloader')
                    self.real_train_iter = iter(self.real_train_loader)
                    inputs = self.real_train_iter.__next__()
            
            # Move all available tensors to GPU memory
            for key, ipt in inputs.items():
                if type(key) == tuple or key == "depth_gt":
                    inputs[key] = ipt.to(self.device)

            # log less frequently after the first 2000 steps to save time & disk space
            self.step += 1
            self.early_phase = batch_idx % self.opt.log_frequency == 0
            self.mid_phase = False and self.step % self.opt.save_frequency == 0
            self.late_phase = self.num_total_batch - 1 == batch_idx

            outputs, losses = {}, {}
            # Depth estimation
            outputs_d, losses_d = self.process_batch(inputs)
            outputs.update(outputs_d)
            losses.update(losses_d)

            # No more if else conditions, just combine all losses based on availability of gradients
            final_loss = torch.tensor(0.).to(self.device)
            for k, v in losses.items():
                if ('d_' not in k) and v.requires_grad and ('/' not in k):
                    final_loss += v
            final_loss.backward()
            losses["loss"] = final_loss

            if (batch_idx + 1) % 2 == 0:
                self.model_optimizer.step()
                self.model_optimizer.zero_grad()
                self.zero_grad()

            if (batch_idx-200) % 400 == 0 or (batch_idx-401) % 400 == 0:
                duration = time.time() - before_op_time
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            # if self.early_phase or self.mid_phase or self.late_phase:
            if (batch_idx-200) % 400 == 0 or (batch_idx-401) % 400 == 0:
                self.log("train", inputs, outputs, losses)
                # self.val("real")
                # self.val("syn")

            if (batch_idx + 1) % 2 == 0:
                self.current_lr = self.model_lr_scheduler.get_last_lr()[0]
                # self.current_lr = self.update_learning_rate(self.model_optimizer, self.opt.learning_rate)

        mean_night_err = self.evaluate(split='night')
        mean_day_err = self.evaluate(split='day')

    def get_lambda(self, epoch, max_epoch):
        p = epoch / max_epoch
        return 2. / (1+np.exp(-10.*p)) - 1.

    # Depth Maps, Semantic Segmentation and Relative Pose Estimation
    def process_batch(self, inputs):
        """Pass a mini-batch through the network and generate images and losses
        """
        # Otherwise, we only feed the image with frame_id 0 through the depth encoder
        features, raw_hrnet_features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth_decoder"](features)

        if self.opt.use_dc:
            self.lambda_ = self.get_lambda(self.epoch, self.opt.num_epochs)
            outputs['domain_classifier'] = self.models['domain_classifier'](raw_hrnet_features, self.lambda_)

        if self.opt.use_pose_net:
            outputs.update(self.predict_poses(inputs, features, self.syn_or_real))

        # convert estimated disparity from neural network to depth
        self.generate_images_pred(inputs, outputs)

        # loss functions
        losses = self.compute_losses_local(inputs, outputs)

        return outputs, losses

    def val(self, subset):
        """Validate the model on a single mini-batch
        """
        self.set_eval()
        if "syn" in subset:
            try:
                inputs = self.syn_val_iter.__next__()
            except StopIteration:
                print('Stopped as the iteration has reached to the END, and reloading the dataloader')
                self.syn_val_iter = iter(self.syn_val_loader)
                inputs = self.syn_val_iter.__next__()
            self.syn_or_real = 'syn'
        elif "real" in subset:
            try:
                inputs = self.real_val_iter.__next__()
            except StopIteration:
                print('Stopped as the iteration has reached to the END, and reloading the dataloader')
                self.real_val_iter = iter(self.real_val_loader)
                inputs = self.real_val_iter.__next__()
            self.syn_or_real = 'real'
        else:
            raise RuntimeError("Need proper validation loader choose syn or real or real_eigen")

        # Move all available tensors to GPU memory
        for key, ipt in inputs.items():
            if type(key) == tuple or key == "depth_gt":
                inputs[key] = ipt.to(self.device)

        outputs, losses = {}, {}
        with torch.no_grad():
            # Estimated depth and segmentation
            outputs_d, losses_d = self.process_batch(inputs)
            outputs.update(outputs_d)
            losses.update(losses_d)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val_" + self.syn_or_real, inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def evaluate(self, split='night'):
        """Evaluates a pretrained model using a specified test set
        """
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80

        assert sum((self.opt.eval_mono, self.opt.eval_stereo)) == 1, \
            "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

        encoder = self.models["encoder"]
        depth_decoder = self.models["depth_decoder"]

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []
        gt = []

        if split=='day':
            dataloader = self.syn_val_loader
            val_split = 'val_day'
        elif split =='night':
            dataloader = self.real_val_loader
            val_split = 'val_night'

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if self.opt.post_process:
                    print("post_process")
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                features, raw_hrnet_features = encoder(input_color)
                output = depth_decoder(features)

                # output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if self.opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = self.batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                gt.append(np.squeeze(data['depth_gt'].cpu().numpy()))

        pred_disps = np.concatenate(pred_disps)
        gt = np.concatenate(gt)

        print("="*50)
        print("-> Evaluating " + val_split)
        print("   Mono evaluation - using median scaling")

        errors = []
        ratios = []

        for i in range(pred_disps.shape[0]):

            gt_depth = gt[i]
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            mask = gt_depth > 0

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            pred_depth *= self.opt.pred_depth_scale_factor
            if not self.opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            # range 60m
            mask2 = gt_depth<=40
            pred_depth = pred_depth[mask2]
            gt_depth = gt_depth[mask2]

            errors.append(self.compute_errors(gt_depth, pred_depth))

        if not self.opt.disable_median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

        mean_errors = np.array(errors).mean(0)

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
        # print("\n-> Done!")

        return mean_errors

    def generate_images_pred_local(self, inputs, outputs):
        if self.syn_or_real == "syn":
            for scale in self.opt.scales:
                disp = outputs[("disp", scale)]
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

                _, depth =  (disp, self.opt.min_depth, self.opt.max_depth)

                outputs[("depth", 0, scale)] = depth

        elif self.syn_or_real == "real":
            self.generate_images_pred(inputs, outputs)
        else:
            raise RuntimeError("choose synthetic or real data")

    def generate_gan_outputs(self, day_disp, night_disp):
        # (n, 1, h, w)
        # remove scale
        night_disp = night_disp / night_disp.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
        day_disp = day_disp / day_disp.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
        # image coordinates
        if self.opt.use_position_map:
            n = night_disp.shape[0]
            height_map = self.height_map.repeat(n, 1, 1, 1).to(self.device)
            width_map = self.width_map.repeat(n, 1, 1, 1).to(self.device)
        else:
            height_map = None
            width_map = None
        return day_disp, night_disp, height_map, width_map
    
    def compute_G_loss(self, night_disp, height_map, width_map):
        G_loss = 0.0
        #
        # Compute G loss
        #
        freeze_model(self.models["depth_classifier"])
        if self.opt.use_position_map:
            fake_day = torch.cat([height_map, width_map, night_disp], dim=1)
        else:
            fake_day = night_disp
        G_loss += self.gan_loss(self.models["depth_classifier"](fake_day), True)

        return G_loss

    def compute_D_loss(self, day_disp, night_disp, height_map, width_map):
        D_loss = 0.0
        #
        # Compute D loss
        #
        unfreeze_model(self.models["depth_classifier"])
        if self.opt.use_position_map:
            real_day = torch.cat([height_map, width_map, day_disp.detach()], dim=1)
            fake_day = torch.cat([height_map, width_map, night_disp.detach()], dim=1)
        else:
            real_day = day_disp.detach()
            fake_day = night_disp.detach()
        # query
        # fake_day = self.image_pool.query(fake_day)
        # compute loss
        D_loss += self.gan_loss(self.models["depth_classifier"](real_day), True)
        D_loss += self.gan_loss(self.models["depth_classifier"](fake_day), False)

        return D_loss

    def compute_losses_local(self, inputs, outputs):
        # Choose loss function based on input data
        self_loss, l1_loss = self.torch_zero.clone(), self.torch_zero.clone()
        losses = {"loss": self.torch_zero.clone()}

        # Domain classifier with Gradient Reversal Layer
        if self.opt.use_dc:
            if self.syn_or_real == 'syn':
                losses['domain_classifier'] = self.lambda_ * self.LossDomainClassifier(outputs['domain_classifier'],
                                                                               self.False_.long())
                ### scale loss
                # losses['domain_classifier'] = 10*self.lambda_ * self.LossDomainClassifier(outputs['domain_classifier'],
                #                                                                self.False_.long())                                                                          
            else:
                losses['domain_classifier'] = self.lambda_ * self.LossDomainClassifier(outputs['domain_classifier'],
                                                                               self.True_.long())
                ### scale loss
                # losses['domain_classifier'] = 10*self.lambda_ * self.LossDomainClassifier(outputs['domain_classifier'],
                #                                                                self.True_.long())
            losses['loss/domain_classifier'] = losses['domain_classifier'].detach().cpu().data

        if 'self' in self.opt.real_loss_fcn and "real" in self.syn_or_real:
            losses.update(self.compute_losses(inputs, outputs))
            # Loss equalizer
            losses["loss/self_real"] = losses["loss"].cpu().data
            weight_self_loss = self.previous_sf_day_loss / losses["loss/self_real"] if self.opt.use_le else self.opt.self_scaling_factor
            losses["loss"] *= weight_self_loss

        elif 'self' in self.opt.syn_loss_fcn and "syn" in self.syn_or_real:
            losses.update(self.compute_losses(inputs, outputs))

            # Loss equalizer
            losses["loss/self_syn"] = losses["loss"].cpu().data
            self.previous_sf_day_loss = losses["loss"].cpu().data
        else:
            raise RuntimeError('choose a loss function for each syn and real dataset')

        if self.epoch >= 15:
            # Loss depth classifier
            if self.syn_or_real == 'syn':
                self.day_disp = outputs['disp', 0]

            elif self.syn_or_real == 'real':
                self.night_disp = outputs['disp', 0]

                # generate outputs for gan
                day_disp, night_disp, height_map, width_map = self.generate_gan_outputs(self.day_disp, self.night_disp)
                #
                # optimize D
                # 
                # compute loss
                D_loss = self.compute_D_loss(day_disp, night_disp, height_map, width_map)
                losses['loss/D_depth_classifier'] = D_loss.detach().cpu().data
                D_loss = D_loss * self.opt.D_weight
                # print(losses['loss/D_depth_classifier'])

                # optimize D
                self.depth_classifier_optim.zero_grad()
                D_loss.backward()
                self.depth_classifier_optim.step()

                #
                # optimize G
                #
                # compute loss
                G_loss = self.compute_G_loss(night_disp, height_map, width_map)
                losses['loss/G_depth_classifier'] = G_loss.detach().cpu().data
                G_loss = G_loss * self.opt.G_weight 
                # print(losses['loss/G_depth_classifier'])

                # optimize G
                losses["G_depth_classifier"] = G_loss

        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_gt = inputs["depth_gt"]
        mask = (depth_gt > self.opt.min_depth) & (depth_gt < self.opt.max_depth)

        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, depth_gt.shape[2:], mode="bilinear", align_corners=False), self.opt.min_depth,
            self.opt.max_depth)
        depth_pred = depth_pred.detach()

        # # garg/eigen crop
        # crop_mask = torch.zeros_like(mask)
        # crop_mask[:, :, 153:371, 44:1197] = 1
        # mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        median_gt_pred = torch.median(depth_gt) / torch.median(depth_pred)
        depth_pred *= median_gt_pred

        depth_pred = torch.clamp(depth_pred, min=self.opt.min_depth, max=self.opt.max_depth)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            if 'median' in metric:
                losses[metric] = median_gt_pred
            else:
                losses[metric] = depth_errors[i]

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "exp_name {}  \n| dataset: {:>5} | epoch {:>3} | batch {:>6}/{:>6} | " \
                       "examples/s: {:5.1f} | loss: {:.5f} | lambda: {:.5f}| lr: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.log_path.split('/')[-1], self.syn_or_real, self.epoch, batch_idx,
                                  self.num_total_batch, samples_per_sec, loss, self.lambda_, self.current_lr, sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(2, self.opt.batch_size)):  # write a maxmimum of four images
            # for s in self.opt.scales:
            for s in [0]:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "depth_pred_color_{}/{}".format(s, j),
                    normalize_image(self.convert_disp_depth_map(outputs[("disp", s)][j].unsqueeze(0))), self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                # elif not self.opt.disable_automasking:
                #     writer.add_image(
                #         "automask_{}/{}".format(s, j),
                #         outputs["identity_selection/{}".format(s)][j][None, ...], self.step)


    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        # # save code as another folder in log_path
        # dst_path = os.path.join(self.log_path, 'code', 'v0')
        # iter_yes_or_no = 0
        # while os.path.exists(dst_path):
        #     dst_path = os.path.join(self.log_path, 'code', 'v' + str(iter_yes_or_no))
        #     iter_yes_or_no = iter_yes_or_no + 1
        # user_name = expanduser("~")
        # try:
        #     shutil.copytree(os.getcwd(), dst_path, ignore=shutil.ignore_patterns('*.pyc', 'tmp*'))
        # except Exception as e_copytree:
        #     print(e_copytree)

        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w', encoding='utf-8') as f:
            json.dump(to_save, f, indent=2)

    def update_learning_rate(self, optimizer, base_lr_rate, power=0.9):
        step = max(1, self.step)
        step = int(step / 2)
        new_lr_rate = base_lr_rate * (1 - (float(step / self.num_total_steps) ** power))
        optimizer.param_groups[0]['lr'] = new_lr_rate

        return new_lr_rate

    # Train with - syn - 0, real - 1, both - 2
    def choosing_dataset_to_train_with(self, index):
        if self.opt.train_with == "syn":
            return True
        elif self.opt.train_with == "real":
            return False
        elif self.opt.train_with == "both":
            if index % 2 == 0:
                return True
            else:
                return False
        else:
            print("choose at-least one of the dataset for training")
            sys.exit(0)

    def supervised_loss(self, inputs, outputs):
        l1_loss = self.torch_zero.clone()
        for i in self.opt.scales:
            pred = outputs[("depth", 0, i)]
            gt = F.interpolate(inputs["depth_gt"], size=pred.shape[2:]) / self.opt.syn_scaling_factor

            mask = torch.zeros(gt.shape, dtype=torch.float).to(self.device)
            mask[gt < (self.opt.max_depth - 2) / self.opt.syn_scaling_factor] = 1
            mask[gt == 0] = 0  # while we train with kitti LiDAR GT

            if self.opt.use_ms:
                mask_segm = torch.zeros(gt.shape, dtype=torch.float).to(self.device)
                mask_segm[(inputs[('segm_gt', 0, 0)] == 8)] = 1.0
                mask_segm[(inputs[('segm_gt', 0, 0)] == 9)] = 1.0
                mask_segm[(inputs[('segm_gt', 0, 0)] == 10)] = 1.0
                mask_segm[(inputs[('segm_gt', 0, 0)] == 11)] = 1.0
                mask_segm[(mask_segm == 0)] = 0.5
                mask = mask * mask_segm

            l1_loss += torch.mean(torch.abs(pred * mask - gt * mask))

        return l1_loss

    def log_evaluation(self, mode, errors_rsf, med_std, errors_asf):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]

        writer.add_scalar("median/mean_median", med_std[0], self.epoch)
        writer.add_scalar("median/median_std", med_std[1], self.epoch)

        writer.add_scalar("depth_rsf/abs_rel", errors_rsf[0], self.epoch)
        writer.add_scalar("depth_rsf/sq_rel", errors_rsf[1], self.epoch)
        writer.add_scalar("depth_rsf/rmse", errors_rsf[2], self.epoch)
        writer.add_scalar("depth_rsf/rmse_log", errors_rsf[3], self.epoch)
        writer.add_scalar("depth_rsf/a1", errors_rsf[4], self.epoch)
        writer.add_scalar("depth_rsf/a2", errors_rsf[5], self.epoch)
        writer.add_scalar("depth_rsf/a3", errors_rsf[6], self.epoch)
        writer.add_scalar("depth_asf/abs_rel", errors_asf[0], self.epoch)
        writer.add_scalar("depth_asf/sq_rel", errors_asf[1], self.epoch)
        writer.add_scalar("depth_asf/rmse", errors_asf[2], self.epoch)
        writer.add_scalar("depth_asf/rmse_log", errors_asf[3], self.epoch)
        writer.add_scalar("depth_asf/a1", errors_asf[4], self.epoch)
        writer.add_scalar("depth_asf/a2", errors_asf[5], self.epoch)
        writer.add_scalar("depth_asf/a3", errors_asf[6], self.epoch)

    # need to keep on updating based on requirement
    # Case sensitive
    def network_selection(self, model_key):
        if model_key == 'encoder':
            if 'HRNet' == self.opt.models_fcn_name[model_key]:
                with open(os.path.join('configs', 'hrnet_w' + str(self.opt.num_layers) + '_vk2.yaml'), 'r') as cfg:
                    config = yaml.safe_load(cfg)
                return networks.HRNetPyramidEncoder(config).to(self.device)
            elif 'DenseNet' == self.opt.models_fcn_name[model_key]:
                return networks.DensenetPyramidEncoder(densnet_version=self.opt.num_layers).to(self.device)
            elif 'ResNet' == self.opt.models_fcn_name[model_key]:
                return networks.ResnetEncoder(self.opt.num_layers,
                                              self.opt.weights_init == "pretrained").to(self.device)
            else:
                raise RuntimeError('Choose a depth encoder within available scope')

        elif model_key == 'depth_decoder':
            if 'DepthDecoder' == self.opt.models_fcn_name[model_key]:
                return networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales).to(self.device)
            else:
                raise RuntimeError('Choose depth Decoder within available scope')

        # Add multiple domain classifiers
        elif model_key == 'domain_classifier':
            inchannels_l, width_l, height_l = self.domain_classifier_input_details()
            return networks.DomainClassifier(in_channel=inchannels_l, width=int(width_l), height=int(height_l)). \
                to(self.device)

        elif model_key == 'pose_encoder':
            return networks.ResnetEncoder(self.opt.num_layers if self.opt.num_layers == 18 else 50,
                                          self.opt.weights_init == "pretrained",
                                          num_input_images=2).to(self.device)

        elif model_key == 'depth_classifier':
            in_chs_D = 3 if self.opt.use_position_map else 1
            return networks.NLayerDiscriminator(in_chs_D, n_layers=3).to(self.device)

        # Add other models
        elif model_key == 'pose':
            if self.opt.pose_model_type == "separate_resnet":
                return networks.PoseDecoder(
                    self.models["pose_encoder_real"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2).to(self.device)

            elif self.opt.pose_model_type == "shared":
                return networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames).to(self.device)

            elif self.opt.pose_model_type == "posecnn":
                return networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2).to(self.device)
            else:
                raise RuntimeError('Choose a depth encoder within available scope')

        else:
            raise RuntimeError('Don\'t forget to mention what you want!')

    def domain_classifier_input_details(self):
        w, h = self.opt.width, self.opt.height
        dc_in_dict = {'DenseNet': {121: [1024, w / 8, h / 8], 169: [1664, w / 8, h / 8], 201: [1920, w / 8, h / 8],
                                   161: [2208, w / 8, h / 8]},
                      'ResNet': {18: [512, w / 8, h / 8], 34: [512, w / 8, h / 8], 50: [2048, w / 8, h / 8],
                                 101: [2048, w / 8, h / 8], 152: [2048, w / 8, h / 8]},
                      'HRNet': {18: [270, w, h], 32: [480, w, h], 48: [720, w, h]}}
        return dc_in_dict[self.opt.models_fcn_name['encoder']][self.opt.num_layers]


if __name__ == "__main__":
    # Load options
    opts = MonoDEVSOptions().parse()

    # Load MonoDEVSNet trainer scripts and start training
    monodevs = MonoDEVSNetTrainer(options=opts)
    monodevs.train()

    TheEnd = 1


