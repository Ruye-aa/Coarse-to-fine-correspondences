import os, math
import numpy as np
import torch
import nibabel.quaternions as nq

def read_trajectory(filename, dim=4):
    with open(filename) as f:
        lines = f.readlines()

        # Extract the point cloud pairs
        keys = lines[0::(dim + 1)]
        temp_keys = []
        for i in range(len(keys)):
            temp_keys.append(keys[i].split('\t')[0:3])

        final_keys = []
        for i in range(len(temp_keys)):
            final_keys.append([temp_keys[i][0].strip(), temp_keys[i][1].strip(), temp_keys[i][2].strip()])

        traj = []
        for i in range(len(lines)):
            if i % 5 != 0:
                traj.append(lines[i].split('\t')[0:dim])

        traj = np.asarray(traj, dtype=float).reshape(-1, dim, dim)

        final_keys = np.asarray(final_keys)

    return final_keys, traj

def read_trajectory_info(filename, dim=6):
    with open(filename) as fid:
        contents = fid.readlines()
    n_pairs = len(contents) // 7
    assert (len(contents) == 7 * n_pairs)
    info_list = []
    n_frame = 0

    for i in range(n_pairs):
        frame_idx0, frame_idx1, n_frame = [int(item) for item in contents[i * 7].strip().split()]
        info_matrix = np.concatenate(
            [np.fromstring(item, sep='\t').reshape(1, -1) for item in contents[i * 7 + 1:i * 7 + 7]], axis=0)
        info_list.append(info_matrix)

    cov_matrix = np.asarray(info_list, dtype=float).reshape(-1, dim, dim)

    return n_frame, cov_matrix

def write_est_trajectory(gt_folder, exp_dir, tsfm_est):
    scene_names=sorted(os.listdir(gt_folder))
    count=0
    for scene_name in scene_names:
        gt_pairs, gt_traj = read_trajectory(os.path.join(gt_folder,scene_name,'gt.log'))
        est_traj = []
        for i in range(len(gt_pairs)):
            est_traj.append(tsfm_est[count])
            count+=1

        # write the trajectory
        c_directory=os.path.join(exp_dir,scene_name)
        os.makedirs(c_directory,exist_ok=True)
        write_trajectory(np.array(est_traj),gt_pairs,os.path.join(c_directory, 'est.log'))

def write_trajectory(index, traj, metadata, filename, dim=4):
    with open(filename, 'w') as f:
        for idx in index:
            # Only save the transfromation parameters for which the overlap threshold was satisfied
            if metadata[idx][2]:
                p = traj[idx,:,:].tolist()
                f.write('\t'.join(map(str, metadata[idx])) + '\n')
                f.write('\n'.join('\t'.join(map('{0:.12f}'.format, p[i])) for i in range(dim)))
                f.write('\n')

def computeTransformationErr(trans, info):
    t = trans[:3, 3]
    r = trans[:3, :3]
    q = nq.mat2quat(r)
    er = np.concatenate([t, q[1:]], axis=0)
    p = er.reshape(1, 6) @ info @ er.reshape(6, 1) / info[0, 0]

    return p.item()


def translation_error(t1, t2):
    return torch.norm(t1 - t2, dim=(1, 2))

def rotation_error(R1, R2):
    R_ = torch.matmul(R1.transpose(1, 2), R2)
    e = torch.stack([(torch.trace(R_[_, :, :]) - 1) / 2 for _ in range(R_.shape[0])], dim=0).unsqueeze(1)

    # Clamp the errors to the valid range (otherwise torch.acos() is nan)
    e = torch.clamp(e, -1, 1, out=None)

    ae = torch.acos(e)
    pi = torch.Tensor([math.pi])
    ae = 180. * ae / pi.to(ae.device).type(ae.dtype)

    return ae

def extract_corresponding_trajectors(est_pairs, gt_pairs, gt_traj):
    ext_traj = np.zeros((len(est_pairs), 4, 4))

    for est_idx, pair in enumerate(est_pairs):
        pair[2] = gt_pairs[0][2]
        gt_idx = np.where((gt_pairs == pair).all(axis=1))[0]

        ext_traj[est_idx, :, :] = gt_traj[gt_idx, :, :]

    return ext_traj


def evaluate_registration_recall(num_fragment, result, result_pairs, gt_pairs, gt, gt_info):
    gt_mask = np.zeros((num_fragment, num_fragment), dtype=int)
    recall = []

    for idx in range(gt_pairs.shape[0]):
        i = int(gt_pairs[idx, 0])
        j = int(gt_pairs[idx, 1])

        # Only non consecutive pairs are tested
        # if j - i > 1:
        gt_mask[i, j] = idx+1

    n_res = 0
    for idx in range(result_pairs.shape[0]):
        i = int(result_pairs[idx, 0])
        j = int(result_pairs[idx, 1])
        pose = result[idx, :, :]

        if gt_mask[i, j] > 0:
            n_res += 1
            gt_idx = gt_mask[i, j]
            p = computeTransformationErr(np.linalg.inv(gt[gt_idx-1, :, :]) @ pose, gt_info[gt_idx-1, :, :])
            recall.append(p)

    return recall

npoint = 250
benchmark = '3DLoMatch'
gt_folder = f'configs/benchmarks/{benchmark}'
est_folder = f'est_traj/{benchmark}/{npoint}'
x = 0

# 提取内点率数据
inliers_folder = f'{est_folder}/inliers_num.txt'
inliers_num = []
inlier_ratio = []
with open(inliers_folder) as f:
    for eachline in f:
        tmp = eachline.split()
        inlier_ratio.append(tmp[5])
inlier_ratio = list(map(float, inlier_ratio))

with open(inliers_folder) as f:
    inliers = f.readlines()

# 获取文件夹内所有文件名
for eachfile in os.listdir(gt_folder):
    print(f"scene: {eachfile}")
    # 读取矩阵文件中的数据
    gt = f'{gt_folder}/{eachfile}/gt.log'
    gt_info = f'{gt_folder}/{eachfile}/gt.info'
    # gt_overlap = f'{gt_folder}/{eachfile}/gt_overlap.log'
    est = f'{est_folder}/{eachfile}/est.log'

    # 读取矩阵和配对信息
    gt_pairs, gt_traj = read_trajectory(gt, dim=4)
    est_pairs, est_traj = read_trajectory(est, dim=4)
    n_fragments, gt_traj_cov = read_trajectory_info(gt_info)
    x += gt_pairs.shape[0]

    # 通过矩阵获得re/te
    ext_gt_traj = extract_corresponding_trajectors(est_pairs, gt_pairs, gt_traj) # 从gt中找出需要的帧
    re = rotation_error(torch.from_numpy(ext_gt_traj[:, 0:3, 0:3]), torch.from_numpy(est_traj[:, 0:3, 0:3])).cpu().numpy()
    te = translation_error(torch.from_numpy(ext_gt_traj[:, 0:3, 3:4]),torch.from_numpy(est_traj[:, 0:3, 3:4])).cpu().numpy()

    # 计算场景中每帧的recall
    recall = evaluate_registration_recall(n_fragments, est_traj, est_pairs, gt_pairs, gt_traj, gt_traj_cov)

    # 提取每个场景真实重复率
    # with open(gt_overlap) as f:
    #     lines = f.readlines()

    # 找到最好/差的10帧
    recall_best, recall_best_idx = torch.topk(torch.from_numpy(np.array(recall)),10, largest=False)
    recall_worst, recall_worst_idx = torch.topk(torch.from_numpy(np.array(recall)), 10, largest=True)
    inlier_best, inlier_best_idx = torch.topk(torch.from_numpy(np.array(inlier_ratio[(x-gt_pairs.shape[0]):x])),10, largest=True)
    inlier_worst, inlier_worst_idx = torch.topk(torch.from_numpy(np.array(np.array(inlier_ratio[x-gt_pairs.shape[0]:x]))), 10, largest=False)
    # 保存最好/差的10个数据的索引
    filename = f'{est_folder}/{eachfile}/best.log'
    with open(filename, 'w') as f:
        for idx in range(recall_best_idx.shape[0]):
            f.write(f'{recall_best_idx[idx].numpy().tolist()}\n')
        for idx in range(inlier_best_idx.shape[0]):
            f.write(f'{inlier_best_idx[idx].numpy().tolist()}\n')

    filename = f'{est_folder}/{eachfile}/worst.log'
    with open(filename, 'w') as f:
        for idx in range(recall_worst_idx.shape[0]):
            f.write(f'{recall_worst_idx[idx].numpy().tolist()}\n')
        for idx in range(inlier_worst_idx.shape[0]):
            f.write(f'{inlier_best_idx[idx].numpy().tolist()}\n')

    # write_trajectory(np.array(recall_best_idx), est_traj, gt_pairs, os.path.join(est_folder,eachfile, 'best_recall.log'))
    # write_trajectory(np.array(recall_worst_idx), est_traj, gt_pairs, os.path.join(est_folder, eachfile, 'worst_recall.log'))
    # write_trajectory(np.array(inlier_best_idx), est_traj, gt_pairs, os.path.join(est_folder, eachfile, 'best_inlier.log'))
    # write_trajectory(np.array(inlier_worst_idx), est_traj, gt_pairs, os.path.join(est_folder, eachfile, 'worst_inlier.log'))

    # 保存每帧的结果recall,RE/TE,inlier_ratio

    with open(os.path.join(est_folder, eachfile, 'eva.log'), 'w') as f:
        for idx in range(gt_pairs.shape[0]):
            f.write('\t'.join(map(str, gt_pairs[idx])) + '\n')
            f.write(f'recall: {recall[idx]:.4f} \n')
            f.write(f'RE: {re[idx].item():.4f} \t')
            f.write(f'TE: {te[idx].item():.4f} \n')
            f.write(f'{inliers[x-gt_pairs.shape[0]+idx]} \n')

    # 每个场景 优/差 占比
    recall_good = 0
    recall_bad = 0
    inliear_good = 0
    inliear_bad = 0
    for i in range(gt_pairs.shape[0]):
        if recall[i]<0.2:
            recall_good += 1
        elif recall[i]>2:
            recall_bad += 1
        if inlier_ratio[x-i-1] > 0.4:   # 100
            inliear_good += 1
        elif inlier_ratio[x-i-1] < 0.08:   # 20
            inliear_bad += 1
    print(f'recall_good_ratio: {recall_good/gt_pairs.shape[0]:.3f}')
    print(f'recall_bad_ratio: {recall_bad/gt_pairs.shape[0]:.3f}')
    print(f'inliear_good_ratio: {inliear_good / gt_pairs.shape[0]:.3f}')
    print(f'inliear_bad_ratio: {inliear_bad / gt_pairs.shape[0]:.3f} \n')




print("scene_est over")
