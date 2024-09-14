from submodules.loss.L1_Structural_loss import l_structural
from submodules.loss.L2_depth_loss import L2_depth_loss

def total_loss(pseudo_gt_map, gt_lidar, dense_depth):
    structural_loss = l_structural(pseudo_gt_map, dense_depth)
    depth_loss = L2_depth_loss(gt_lidar, dense_depth)

    loss = structural_loss + depth_loss

    return loss, structural_loss, depth_loss