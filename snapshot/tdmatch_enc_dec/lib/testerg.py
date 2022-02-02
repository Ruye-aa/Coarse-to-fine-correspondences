# For train-----

import os, torch
import tqdm
import numpy as np
from lib.utils import get_fine_grained_correspondences, correspondences_from_score_max
from lib.benchmark_utils import ransac_pose_estimation_correspondences, to_array, get_angle_deviation
# modelnet part
from common.math_torch import se3
from common.math.so3 import dcm2euler

import gc
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
from lib.utils import AverageMeter, Logger

import logging

class TDMatchTester():
    '''
    3DMatch Tester
    '''

    def __init__(self, args):
        self.config = args
        ###########################
        # parameters
        ###########################
        self.verbose = args.verbose
        self.verbose_freq = args.verbose_freq
        self.start_epoch = 1
        self.max_epoch = args.max_epoch
        self.training_max_iter = args.training_max_iter
        self.val_max_iter = args.val_max_iter
        #self.device = args.device
        self.best_loss = 1e5
        self.best_matching_recall = -1e5
        self.best_local_matching_precision = -1e5

        self.save_dir = args.save_dir
        self.snapshot_dir = args.snapshot_dir

        self.model = args.model.cuda()
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_interval = args.scheduler_interval
        self.snapshot_interval = args.snapshot_interval
        self.iter_size = args.iter_size

        self.w_matching_loss = args.w_matching_loss
        self.w_local_matching_loss = args.w_local_matching_loss

        self.writer = SummaryWriter(logdir=args.tboard_dir)
        self.logger = Logger(self.snapshot_dir)
        self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()]) / 1000000.} M\n')

        if args.pretrain != '':
            self._load_pretrain(args.pretrain)

        self.loader = dict()

        self.loader['train'] = args.train_loader
        self.loader['val'] = args.val_loader
        self.loader['test'] = args.test_loader
        self.desc_loss = args.desc_loss

        with open(f'{args.snapshot_dir}/model.log', 'w') as f:
            f.write(str(self.model))

        f.close()

    def train(self):
        # logger = logging.getLogger(__name__)
        # logger.info('Start training...')
        print('start training...')
        for epoch in range(self.start_epoch, self.max_epoch):
            ####logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
            # self.inference_one_epoch(epoch, 'train')
            gc.collect()    #清除内存
            #init stats meter
            #stats_meter = self.stats_meter()
            meters = dict()
            #stats = self.stats_dict()
            stats = dict()
            stats['matching_loss'] = 0.
            stats['matching_recall'] = 0.
            stats['local_matching_loss'] = 0.
            stats['local_matching_precision'] = 0.
            stats['total_loss'] = 0.

            for key, _ in stats.items():
                meters[key] = AverageMeter()
            stats_meter = meters

            num_iter = int(len(self.loader['train'].dataset) // self.loader['train'].batch_size)
            c_loader_iter = self.loader['train'].__iter__()

            self.optimizer.zero_grad()      #初始化更新参数
            for c_iter in tqdm(range(num_iter)):
                inputs = c_loader_iter.next()
                for k, v in inputs.items():
                    if type(v) == list:
                        inputs[k] = [item.cuda() for item in v]
                    else:
                        inputs[k] = v.cuda()
                # forward pass
                #stats = self.inference_one_batch(inputs, 'train')
                self.model.train()

                # forward pass
                scores, local_scores, local_scores_gt = self.model.forward(inputs)
                matching_mask = inputs['matching_mask']

                # get loss
                stats = self.desc_loss(scores[0], matching_mask, local_scores, local_scores_gt)
                c_loss = self.w_matching_loss * stats['matching_loss'] + \
                         self.w_local_matching_loss * stats['local_matching_loss']

                c_loss.backward()
                # detach gradients for loss terms
                stats['matching_loss'] = float(stats['matching_loss'].detach())
                stats['local_matching_loss'] = float(stats['local_matching_loss'].detach())
                stats['total_loss'] = float(stats['total_loss'].detach())
                stats['matching_recall'] = stats['matching_recall']
                stats['local_matching_precision'] = stats['local_matching_precision']


                # run optimization
                if (c_iter + 1) % self.iter_size == 0 :
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                # update to stats_meter
                for key, value in stats.items():
                    stats_meter[key].update(value)

                torch.cuda.empty_cache()
                if self.verbose and (c_iter + 1) % self.verbose_freq * 20 == 0:
                    curr_iter = num_iter * (epoch - 1) + c_iter
                    for key, value in stats_meter.items():
                        self.writer.add_scalar(f'{"train"}/{key}', value.avg, curr_iter)

                    message = f'{"train"} Epoch: {epoch} [{c_iter + 1:4d}/{num_iter}]'
                    for key, value in stats_meter.items():
                        message += f'{key}:{value.avg:.2f}\t'

                    self.logger.write(message + '\n')
            message = f'{"train"} Epoch: {epoch}'
            for key, value in stats_meter.items():
                message += f'{key}: {value.avg:.4f}\t'

            self.logger.write(message + '\n')

            #return stats_meter

            self.scheduler.step()    #根据梯度更新网络参数

            #stats_meter = self.inference_one_epoch(epoch, 'val')
            #-----------------------------------------------------------------------
            gc.collect()    #清除内存
            #init stats meter
            stats_meter = self.stats_meter()

            num_iter = int(len(self.loader['val'].dataset) // self.loader['val'].batch_size)
            c_loader_iter = self.loader['val'].__iter__()

            self.optimizer.zero_grad()      #初始化更新参数

            for c_iter in tqdm(range(num_iter)):
                inputs = c_loader_iter.next()
                for k, v in inputs.items():
                    if type(v) == list:
                        inputs[k] = [item.cuda() for item in v]
                    else:
                        inputs[k] = v.cuda()

                # forward pass
                stats = self.inference_one_batch(inputs, 'val')

                # update to stats_meter
                for key, value in stats.items():
                    stats_meter[key].update(value)

                torch.cuda.empty_cache()

            message = f'{"val"} Epoch: {epoch}'
            for key, value in stats_meter.items():
                message += f'{key}: {value.avg:.4f}\t'

            self.logger.write(message + '\n')
            #-------------------------------------------------------------------------------            '''
            if stats_meter['total_loss'].avg < self.best_loss:
                self.best_loss = stats_meter['total_loss'].avg
                self._snapshot(epoch, 'best_loss')

            if stats_meter['local_matching_precision'].avg > self.best_local_matching_precision:
                self.best_local_matching_precision = stats_meter['local_matching_precision'].avg
                self._snapshot(epoch, 'best_local_matching_precision')

            if stats_meter['matching_recall'].avg > self.best_matching_recall:
                self.best_matching_recall = stats_meter['matching_recall'].avg
                self._snapshot(epoch, 'best_matching_recall')

        # finish all epoch
        print('training finish!')