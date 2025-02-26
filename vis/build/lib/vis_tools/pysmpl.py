from .smpl.SMPL import SMPL_layer
import numpy as np
import torch
import torch.nn as nn
import roma
import os


class PySMPL(nn.Module):
    def __init__(self):
        super().__init__()
        h36m_jregressor = np.load(os.path.join(os.path.dirname(__file__), "data/smpl/J_regressor_h36m.npy"))
        feet_jregressor = np.load(os.path.join(os.path.dirname(__file__), "data/smpl/J_regressor_feet.npy"))
        wham_jregressor = np.load(os.path.join(os.path.dirname(__file__), "data/smpl/J_regressor_wham.npy"))
        self.smpl_dtype = torch.float32
        self.num_joints = 24
        self.smpl = SMPL_layer(
                os.path.join(os.path.dirname(__file__),
            "data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"),
            h36m_jregressor=h36m_jregressor,
            feet_jregressor=feet_jregressor,
            wham_jregressor=wham_jregressor,
            dtype=self.smpl_dtype,
            num_joints=self.num_joints,
        )
        self.joint_pairs_24 = (
            (1, 2),
            (4, 5),
            (7, 8),
            (10, 11),
            (13, 14),
            (16, 17),
            (18, 19),
            (20, 21),
            (22, 23),
        )
        self.leaf_pairs = ((0, 1), (3, 4))
        self.root_idx_24 = 0
        self.faces = self.smpl.faces
        self.parents = self.smpl.parents
        self.weights = self.smpl.lbs_weights
        self.eval()

    def forward(self, _betas, _pose, _transl=None):
        betas = _betas.view(-1, 10)
        pose = _pose.view(-1, 24, 3)
        if _transl is not None:
            transl = _transl.view(-1, 3)
        else:
            transl = _transl
        global_pose = pose[..., :1, :]
        local_pose = pose[..., 1:, :]
        output = self.smpl(local_pose, betas, global_pose, transl)
        return output

    def get_phi(self, shape, pose):
        rotmat = roma.rotvec_to_rotmat(pose).view(-1, 24, 3, 3)
        betas = shape.view(-1, 10)
        phi = self.smpl.get_phi(betas, rotmat)
        return phi

    def get_leaf_rotvec(self, x):
        _x = x.clone()
        mask = torch.zeros_like(x).bool()
        mask[..., [15, 10, 11, 22, 23], :] = True
        _x[~mask] = 0
        return _x

    def get_parents(self):
        return self.smpl.parents

    def get_globalR(self, shape, kp3d_rel):
        global_pose = kp3d_rel[..., :1, :] * 0
        local_pose = kp3d_rel[..., 1:, :] * 0
        betas = shape.view(-1, 10)
        output = self.smpl(local_pose, betas, global_pose, None)
        return self.smpl.get_global_pose(kp3d_rel, output.joints)


if __name__ == "__main__":
    _smpl = PySMPL()
    x = torch.randn(2, 24, 3)
    print(_smpl.get_leaf_rotvec(x))
