#from lib.trainer import Trainer
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
    def _snapshot(self, epoch, name=None):
        '''
        Save a trained model
        :param epoch: index of epoch of current model
        :param name: path to the saving model
        :return: None
        '''
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_matching_recall': self.best_matching_recall,
            'best_local_matching_precision': self.best_local_matching_precision
        }
        if name is None:
            filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')

        print(f'Save model to {filename}')
        self.logger.write(f'Save model to {filename}\n')
        torch.save(state, filename)

    def _load_pretrain(self, resume):
        '''
        Load a pretrained model
        :param resume: the path to the pretrained model
        :return: None
        '''
        if os.path.isfile(resume):
            print(f'=> loading checkpoint {resume}')
            state = torch.load(resume)
            self.start_epoch = state['epoch']
            self.model.load_state_dict(state['state_dict'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.best_loss = state['best_loss']
            self.best_matching_recall = state['best_matching_recall']
            self.best_local_matching_precision = state['best_local_matching_precision']

            self.logger.write(f'Successfully load pretrained model from {resume}!\n')
            self.logger.write(f'Current best loss {self.best_loss}\n')
            self.logger.write(f'Current best matching recall {self.best_matching_recall}\n')
            self.logger.write(f'Current best local matching precision {self.best_local_matching_precision}\n')

        else:
            raise ValueError(f'=> no checkpoint found at {resume}')

    def _get_lr(self, group=0):
        '''
        Get current learning rate
        :param group:
        :return: None
        '''
        return self.optimizer.param_groups[group]['lr']

    def stats_dict(self):
        '''
        Create the dict of all metrics
        :return: stats: the dict containing all metrics
        '''
        stats = dict()
        stats['matching_loss'] = 0.
        stats['matching_recall'] = 0.
        stats['local_matching_loss'] = 0.
        stats['local_matching_precision'] = 0.
        stats['total_loss'] = 0.
        '''
        to be added
        '''
        return stats

    def stats_meter(self):
        '''
        For each metric in stats dict, create an AverageMeter() for update
        :return: meters: dict of AverageMeter()
        '''
        meters = dict()
        stats = self.stats_dict()
        for key, _ in stats.items():
            meters[key] = AverageMeter()
        return meters

    def inference_one_batch(self, input_dict, phase):
        '''

        :param input_dict:
        :param phase:
        :return:
        '''
        assert phase in ['train', 'val', 'test']
        #############################################
        # training
        if (phase == 'train'):
            self.model.train()
            ###############
            # forward pass
            ###############
            scores, local_scores, local_scores_gt = self.model.forward(input_dict)
            matching_mask = input_dict['matching_mask']

            ###############
            # get loss
            ###############

            stats = self.desc_loss(scores[0], matching_mask, local_scores, local_scores_gt)
            c_loss = self.w_matching_loss * stats['matching_loss'] + \
                     self.w_local_matching_loss * stats['local_matching_loss']

            c_loss.backward()
        else:
            self.model.eval()
            with torch.no_grad():
                ###############
                # forward pass
                ###############

                matching_mask = input_dict['matching_mask']

                scores, local_scores, local_scores_gt = self.model.forward(input_dict)

                ###############
                # get loss
                ###############

                stats = self.desc_loss(scores[0], matching_mask, local_scores, local_scores_gt)

        ######################################
        # detach gradients for loss terms
        ######################################
        stats['matching_loss'] = float(stats['matching_loss'].detach())
        stats['local_matching_loss'] = float(stats['local_matching_loss'].detach())
        stats['total_loss'] = float(stats['total_loss'].detach())
        stats['matching_recall'] = stats['matching_recall']
        stats['local_matching_precision'] = stats['local_matching_precision']
        return stats

    def inference_one_epoch(self, epoch, phase):
        '''

        :param epoch:
        :param phase:
        :return:
        '''
        gc.collect()    #清除内存
        assert phase in ['train', 'val', 'test']   #选择阶段

        #init stats meter
        stats_meter = self.stats_meter()

        num_iter = int(len(self.loader[phase].dataset) // self.loader[phase].batch_size)
        c_loader_iter = self.loader[phase].__iter__()

        self.optimizer.zero_grad()      #初始化更新参数

        for c_iter in tqdm(range(num_iter)):
            inputs = c_loader_iter.next()
            for k, v in inputs.items():
                if type(v) == list:
                    inputs[k] = [item.cuda() for item in v]
                else:
                    inputs[k] = v.cuda()

            ####################################
            # forward pass
            ####################################
            stats = self.inference_one_batch(inputs, phase)

            ####################################
            # run optimization
            ####################################
            if (c_iter + 1) % self.iter_size == 0 and phase == 'train':
                self.optimizer.step()
                self.optimizer.zero_grad()

            ####################################
            # update to stats_meter
            ####################################
            for key, value in stats.items():
                stats_meter[key].update(value)


            torch.cuda.empty_cache()

            if self.verbose and (c_iter + 1) % self.verbose_freq * 20 == 0:
                curr_iter = num_iter * (epoch - 1) + c_iter
                for key, value in stats_meter.items():
                    self.writer.add_scalar(f'{phase}/{key}', value.avg, curr_iter)

                message = f'{phase} Epoch: {epoch} [{c_iter + 1:4d}/{num_iter}]'
                for key, value in stats_meter.items():
                    message += f'{key}:{value.avg:.2f}\t'

                self.logger.write(message + '\n')

        message = f'{phase} Epoch: {epoch}'
        for key, value in stats_meter.items():
            message += f'{key}: {value.avg:.4f}\t'

        self.logger.write(message + '\n')

        return stats_meter

    def train(self):
        '''

        :return:
        '''
        print('start training...')
        for epoch in range(self.start_epoch, self.max_epoch):
            self.inference_one_epoch(epoch, 'train')
            self.scheduler.step()    #根据梯度更新网络参数

            stats_meter = self.inference_one_epoch(epoch, 'val')

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

    def eval(self):
        print('Start to evaluate on validation datasets...')
        stats_meter = self.inference_one_epoch(0, 'val')

        for key, value in stats_meter.items():
            print(key, value.avg)


    def test(self):
        print('Start to evaluate on test datasets...')
        os.makedirs(f'{self.snapshot_dir}/{self.config.benchmark}', exist_ok=True)

        num_iter = int(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()
        self.model.eval()
        total_corr_num = 0
        with torch.no_grad():
            for idx in tqdm(range(num_iter)):
                inputs = c_loader_iter.next()
                ##############################
                # Load inputs to device
                for k, v in inputs.items():
                    if v is None:
                        pass
                    elif type(v) == list:
                        inputs[k] = [items.to(self.device) for items in v]
                    else:
                        inputs[k] = v.to(self.device)

                #############################
                # Forward pass
                len_src_pcd = inputs['stack_lengths'][0][0]
                pcds = inputs['points'][0]
                src_pcd, tgt_pcd = pcds[:len_src_pcd], pcds[len_src_pcd:]

                len_src_nodes = inputs['stack_lengths'][-1][0]
                nodes = inputs['points'][-1]
                src_node, tgt_node = nodes[:len_src_nodes], nodes[len_src_nodes:]

                rot = inputs['rot']
                trans = inputs['trans']

                src_candidates_c, tgt_candidates_c, local_scores, node_corr, node_corr_conf, src_pcd_sel, tgt_pcd_sel = self.model.forward(inputs)

                total_corr_num += node_corr.shape[0]

                correspondences, corr_conf = get_fine_grained_correspondences(local_scores, mutual=False, supp=False, node_corr_conf=node_corr_conf)

                data = dict()
                data['src_pcd'], data['tgt_pcd'] = src_pcd.cpu(), tgt_pcd.cpu()
                data['src_node'], data['tgt_node'] = src_node.cpu(), tgt_node.cpu()
                data['src_candidate'], data['tgt_candidate'] = src_candidates_c.view(-1, 3).cpu(), tgt_candidates_c.view(-1, 3).cpu()
                data['src_candidate_id'], data['tgt_candidate_id'] = src_pcd_sel.cpu(), tgt_pcd_sel.cpu()
                data['rot'] = rot.cpu()
                data['trans'] = trans.cpu()
                data['correspondences'] = correspondences.cpu()
                data['confidence'] = corr_conf.cpu()

                torch.save(data, f'{self.snapshot_dir}/{self.config.benchmark}/{idx}.pth')

        print(f'Avg Node Correspondences: {total_corr_num/num_iter}')


class KITTITester(Trainer):
    """
    KITTI tester
    """

    def __init__(self, args):
        Trainer.__init__(self, args)

    def test(self):
        print('Start to evaluate on test datasets...')
        tsfm_est = []
        num_iter = int(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()

        self.model.eval()
        rot_gt, trans_gt = [], []
        with torch.no_grad():
            for _ in tqdm(range(num_iter)):  # loop through this epoch
                inputs = c_loader_iter.next()
                ###############################################
                # forward pass
                for k, v in inputs.items():
                    if type(v) == list:
                        inputs[k] = [item.to(self.device) for item in v]
                    else:
                        inputs[k] = v.to(self.device)

                len_src_nodes = inputs['stack_lengths'][-1][0]
                nodes = inputs['points'][-1]

                c_rot, c_trans = inputs['rot'], inputs['trans']
                rot_gt.append(c_rot.cpu().numpy())
                trans_gt.append(c_trans.cpu().numpy())

                src_candidates_c, tgt_candidates_c, local_scores, node_corr, node_corr_conf, src_pcd_sel, tgt_pcd_sel = self.model.forward(inputs)
                correspondences, corr_conf = get_fine_grained_correspondences(local_scores, mutual=False, supp=False, node_corr_conf=node_corr_conf)
                ########################################
                # run probabilistic sampling
                n_points = 5000

                if (correspondences.shape[0] > n_points):
                    idx = np.arange(correspondences.shape[0])
                    prob = corr_conf.squeeze(1)
                    prob = prob / prob.sum()
                    idx = np.random.choice(idx, size=n_points, replace=False, p=prob.cpu().numpy())
                    correspondences = correspondences.cpu().numpy()[idx]

                src_pcd_reg = src_candidates_c.view(-1, 3).cpu().numpy()[correspondences[:, 0]]
                tgt_pcd_reg = tgt_candidates_c.view(-1, 3).cpu().numpy()[correspondences[:, 1]]
                correspondences = torch.arange(end=src_pcd_reg.shape[0]).unsqueeze(-1).repeat(1, 2).numpy()
                ########################################
                # run ransac
                distance_threshold = 0.3
                ts_est = ransac_pose_estimation_correspondences(src_pcd_reg, tgt_pcd_reg, correspondences,
                                                                distance_threshold=distance_threshold, ransac_n=4)
                tsfm_est.append(ts_est)

        tsfm_est = np.array(tsfm_est)
        rot_est = tsfm_est[:, :3, :3]
        trans_est = tsfm_est[:, :3, 3]
        rot_gt = np.array(rot_gt)
        trans_gt = np.array(trans_gt)[:, :, 0]

        rot_threshold = 5
        trans_threshold = 2

        np.savez(f'{self.snapshot_dir}/results', rot_est=rot_est, rot_gt=rot_gt, trans_est=trans_est, trans_gt=trans_gt)

        r_deviation = get_angle_deviation(rot_est, rot_gt)
        translation_errors = np.linalg.norm(trans_est - trans_gt, axis=-1)

        flag_1 = r_deviation < rot_threshold
        flag_2 = translation_errors < trans_threshold
        correct = (flag_1 & flag_2).sum()
        precision = correct / rot_gt.shape[0]

        message = f'\n Registration recall: {precision:.3f}\n'

        r_deviation = r_deviation[flag_1]
        translation_errors = translation_errors[flag_2]

        errors = dict()
        errors['rot_mean'] = round(np.mean(r_deviation), 3)
        errors['rot_median'] = round(np.median(r_deviation), 3)
        errors['trans_rmse'] = round(np.mean(translation_errors), 3)
        errors['trans_rmedse'] = round(np.median(translation_errors), 3)
        errors['rot_std'] = round(np.std(r_deviation), 3)
        errors['trans_std'] = round(np.std(translation_errors), 3)

        message += str(errors)
        print(message)
        self.logger.write(message + '\n')


def compute_metrics(data , pred_transforms):
    """
    Compute metrics required in the paper
    """
    def square_distance(src, dst):
        return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)

    with torch.no_grad():
        pred_transforms = pred_transforms
        gt_transforms = data['transform_gt']
        points_src = data['points_src'][..., :3]
        points_ref = data['points_ref'][..., :3]
        points_raw = data['points_raw'][..., :3]

        # Euler angles, Individual translation errors (Deep Closest Point convention)
        # TODO Change rotation to torch operations
        r_gt_euler_deg = dcm2euler(gt_transforms[:, :3, :3].numpy(), seq='xyz')
        r_pred_euler_deg = dcm2euler(pred_transforms[:, :3, :3].numpy(), seq='xyz')
        t_gt = gt_transforms[:, :3, 3]
        t_pred = pred_transforms[:, :3, 3]
        r_mse = np.mean((r_gt_euler_deg - r_pred_euler_deg) ** 2, axis=1)
        r_mae = np.mean(np.abs(r_gt_euler_deg - r_pred_euler_deg), axis=1)
        t_mse = torch.mean((t_gt - t_pred) ** 2, dim=1)
        t_mae = torch.mean(torch.abs(t_gt - t_pred), dim=1)

        # Rotation, translation errors (isotropic, i.e. doesn't depend on error
        # direction, which is more representative of the actual error)
        concatenated = se3.concatenate(se3.inverse(gt_transforms), pred_transforms)
        rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
        residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
        residual_transmag = concatenated[:, :, 3].norm(dim=-1)

        # Modified Chamfer distance
        src_transformed = se3.transform(pred_transforms, points_src)
        ref_clean = points_raw
        src_clean = se3.transform(se3.concatenate(pred_transforms, se3.inverse(gt_transforms)), points_raw)
        dist_src = torch.min(square_distance(src_transformed, ref_clean), dim=-1)[0]
        dist_ref = torch.min(square_distance(points_ref, src_clean), dim=-1)[0]
        chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

        metrics = {
            'r_mse': r_mse,
            'r_mae': r_mae,
            't_mse': to_array(t_mse),
            't_mae': to_array(t_mae),
            'err_r_deg': to_array(residual_rotdeg),
            'err_t': to_array(residual_transmag),
            'chamfer_dist': to_array(chamfer_dist)
        }

    return metrics


def print_metrics(logger, summary_metrics , losses_by_iteration=None,title='Metrics'):
    """Prints out formated metrics to logger"""

    logger.info(title + ':')
    logger.info('=' * (len(title) + 1))

    if losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.5f}'.format(c) for c in losses_by_iteration])
        logger.info('Losses by iteration: {}'.format(losses_all_str))

    logger.info('DeepCP metrics:{:.4f}(rot-rmse) | {:.4f}(rot-mae) | {:.4g}(trans-rmse) | {:.4g}(trans-mae)'.format(
        summary_metrics['r_rmse'], summary_metrics['r_mae'],
        summary_metrics['t_rmse'], summary_metrics['t_mae'],
    ))
    logger.info('Rotation error {:.4f}(deg, mean) | {:.4f}(deg, rmse)'.format(
        summary_metrics['err_r_deg_mean'], summary_metrics['err_r_deg_rmse']))
    logger.info('Translation error {:.4g}(mean) | {:.4g}(rmse)'.format(
        summary_metrics['err_t_mean'], summary_metrics['err_t_rmse']))
    logger.info('Chamfer error: {:.7f}(mean-sq)'.format(
        summary_metrics['chamfer_dist']
    ))

def summarize_metrics(metrics):
    """Summaries computed metrices by taking mean over all data instances"""
    summarized = {}
    for k in metrics:
        if k.endswith('mse'):
            summarized[k[:-3] + 'rmse'] = np.sqrt(np.mean(metrics[k]))
        elif k.startswith('err'):
            summarized[k + '_mean'] = np.mean(metrics[k])
            summarized[k + '_rmse'] = np.sqrt(np.mean(metrics[k]**2))
        else:
            summarized[k] = np.mean(metrics[k])

    return summarized


def get_trainer(config):
    '''
    :param config:
    :return:
    '''

    if config.dataset == 'tdmatch':
        return TDMatchTester(config)
    # elif config.dataset == 'kitti':
    #     return KITTITester(config)
    else:
        raise NotImplementedError
