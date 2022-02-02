import os, math, glob
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

        traj = np.asarray(traj, dtype=np.float).reshape(-1, dim, dim)

        final_keys = np.asarray(final_keys)

        return final_keys, traj

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

def write_trajectory(traj, metadata, filename, dim=4):
    with open(filename, 'w') as f:
        for idx in range(traj.shape[0]):
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


npoint = 250
benchmark = '3DMatch'
gt_folder = f'E:/wenxian/code/Pytorch/Coarse-to-fine-correspondences/configs/benchmarks/{benchmark}'
est_folder = f'E:/wenxian/code/Pytorch/Coarse-to-fine-correspondences/est_traj_f/{benchmark}/{npoint}'
feats_scores = sorted(glob.glob(f'{gt_folder}/*'))
print("Mean correspondence numbers")
