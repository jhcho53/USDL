import torch
from torch.nn import Module
from submodules.DCU import depthCompletionNew_blockN
from submodules.data_rectification import rectify_depth

class DenseLiDAR(Module):
    def __init__(self, bs):
        super().__init__()
        self.bs = bs
        # self.rectification = rectify_depth
        self.DCU = depthCompletionNew_blockN(bs)

    def forward(self, image, sparse, device):
        # rectified_depth = self.rectification(sparse, pseudo_depth_map)
        normal2 = self.DCU(image, sparse)
        
        # residual = normal2 - sparse
        # residual = torch.clamp(residual, min=0)
        final_dense_depth = normal2

        return final_dense_depth
    