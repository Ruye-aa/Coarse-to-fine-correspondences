import open3d as o3d
import numpy as np
import copy, torch
import os
import sys

def to_o3d_pcd(pts):
    '''
    From numpy array, make point cloud in open3d format
    :param pts: point cloud (nx3) in numpy array
    :return: pcd: point cloud in open3d format
    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def trans(node, traj):
    rot = traj[:3, :3]
    trans = traj[:3, 2:3]
    node = (torch.matmul(rot, node.T) + trans).T
    return node

def get_inlier_num(src_node, tgt_node, traj, inlier_distance_threshold=0.1):
    '''
    Compute inlier ratios based on input torch tensors
    '''
    rot = traj[:3, :3]
    trans = traj[:3, 2:3]
    src_node = (torch.matmul(rot, src_node.T) + trans).T
    dist = torch.norm(src_node - tgt_node, dim=-1)
    inliers = dist < inlier_distance_threshold
    inliers_num = torch.sum(inliers)
    return inliers_num

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

def extract_corresponding_trajectors(est_pairs, gt_pairs, gt_traj):
    ext_traj = np.zeros((len(est_pairs), 4, 4))

    for est_idx, pair in enumerate(est_pairs):
        pair[2] = gt_pairs[0][2]
        gt_idx = np.where((gt_pairs == pair).all(axis=1))[0]

        ext_traj[est_idx, :, :] = gt_traj[gt_idx, :, :]

    return ext_traj

benchmark = '3DLoMatch'
npoint = 250     # ----------------------
data_folder = f'scripts/data/indoor/test'
gt_folder = f'configs/benchmarks/{benchmark}'
est_folder = f'est_traj/{benchmark}/{npoint}'
inliers_eva = f'{est_folder}/inliers_eva'
x = 0

inliers = f'{est_folder}/inliers_num.txt'
inliers_num = []
with open(inliers) as f:
    for eachline in f:
        tmp = eachline.split()
        inliers_num.append(tmp[3])
inlier_num = list(map(int, inliers_num))

for each_scene in os.listdir(data_folder):
    print(f"scene: {each_scene}")
    gt = f'{gt_folder}/{each_scene}/gt.log'
    est = f'{est_folder}/{each_scene}/est.log'
    gt_pairs, gt_traj = read_trajectory(gt, dim=4)
    est_pairs, est_traj = read_trajectory(est, dim=4)

    # 获取配准帧的索引序号
    best_idx, worst_idx = [], []
    with open(f'{est_folder}/{each_scene}/best.log') as f:
        for eachline in f:
            tmp = eachline.split()
            best_idx.append(int(tmp[0]))     # str
    with open(f'{est_folder}/{each_scene}/worst.log') as f:
        for eachline in f:
            tmp = eachline.split()
            worst_idx.append(int(tmp[0]))
    for i in best_idx:
        src_idx = gt_pairs[i][0]
        tgt_idx = gt_pairs[i][1]
        # 从gt和est中获取对应帧的转换矩阵
        gt_traj_ext = extract_corresponding_trajectors(est_pairs, gt_pairs, gt_traj)[i]
        est_traj_ext = extract_corresponding_trajectors(est_pairs, gt_pairs, est_traj)[i]
        # src和tgt数据
        src_path = f"{data_folder}/{each_scene}/cloud_bin_{src_idx}.pth"
        tgt_path = f"{data_folder}/{each_scene}/cloud_bin_{tgt_idx}.pth"
        tgt_pcd = torch.load(tgt_path)   # 转换为o3d格式
        src_pcd = torch.load(src_path)
        # src和tgt关键点
        tgt_keys_idx = []
        src_keys_idx = []

        with open(f'{inliers_eva}/{x+i}.txt') as f:  # keys_path = f'{inliers_eva}/{i}.txt'
            for eachline in f:
                tmp = eachline.split()
                tgt_keys_idx.append(int(tmp[0]))
                src_keys_idx.append(int(tmp[1]))
        tgt_keys_idx = torch.from_numpy(np.array(tgt_keys_idx))
        src_keys_idx = torch.from_numpy(np.array(src_keys_idx))
        tgt_keys = tgt_pcd[tgt_keys_idx, :]
        src_keys = src_pcd[src_keys_idx, :]
        # inliers_num = get_inlier_num(torch.from_numpy(src_keys), torch.from_numpy(tgt_keys), torch.from_numpy(est_traj_ext))
        tgt_keys = to_o3d_pcd(tgt_keys)
        src_keys = to_o3d_pcd(src_keys)
        # tgt关键点和所有点的转换
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
        src_pcd.paint_uniform_color([1, 0.706, 0])
        tgt_pcd_est = copy.deepcopy(tgt_pcd).transform(est_traj_ext)
        tgt_pcd_gt = copy.deepcopy(tgt_pcd).transform(gt_traj_ext)
        tgt_pcd_est_trans = copy.deepcopy(tgt_pcd_est).translate((0, 4, 0))
        tgt_keys_est = copy.deepcopy(tgt_keys).transform(est_traj_ext)
        tgt_keys_est_trans = copy.deepcopy(tgt_keys_est).translate((0, 4, 0))
        # 关键点合并以及之间的连线
        keys = src_keys + tgt_keys_est_trans
        keys.paint_uniform_color([1, 0, 0])
        corr_keys = np.asarray(keys.points)
        in_colors = [[0, 1, 0] for j in range(inlier_num[i])]
        out_colors = [[1, 0, 0] for j in range(npoint-inlier_num[i])]
        in_lines = [[j, j+npoint] for j in range(inlier_num[i])]
        out_lines = [[inlier_num[i]+j, inlier_num[i]+j+npoint] for j in range(npoint-inlier_num[i])]
        in_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(corr_keys),
            lines=o3d.utility.Vector2iVector(in_lines),
        )
        out_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(corr_keys),
            lines=o3d.utility.Vector2iVector(out_lines),
        )
        in_line_set.colors = o3d.utility.Vector3dVector(in_colors)
        out_line_set.colors = o3d.utility.Vector3dVector(out_colors)
        o3d.visualization.draw([{
            "name": "src",
            "geometry": src_pcd
        }, {
            "name": "tgt_gt",
            "geometry": tgt_pcd_gt
        }, {
            "name": "tgt_est",
            "geometry": tgt_pcd_est
        }, {
            "name": "tgt_est_trans",
            "geometry": tgt_pcd_est_trans
        }, {
            "name": "inlier",
            "geometry": in_line_set
        }, {
            "name": "outlier",
            "geometry": out_line_set
        }, {
            "name": "keys",
            "geometry": keys
        }
        ], show_ui=True)
    x += gt_pairs.shape[0]

