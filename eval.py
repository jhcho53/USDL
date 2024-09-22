import argparse
import os
import torch
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloader import dataLoader as lsn
from dataloader import trainLoader as DA
from model import DenseLiDAR
from tqdm import tqdm
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='depth completion evaluation')
parser.add_argument('--data_path', default='', help='data path')
parser.add_argument('--batch_size', type=int, default=1, help='batch size to evaluate')
parser.add_argument('--gpu_nums', type=int, default=1, help='number of GPUs to evaluate')
parser.add_argument('--loadmodel', default='', help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA evaluation')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--master_addr', type=str, default='localhost', help='master address for distributed evaluation')
parser.add_argument('--master_port', type=str, default='12355', help='master port for distributed evaluation')
args = parser.parse_args()

def setup(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def rmse(gt, pred):
    mask = gt > 0
    return np.sqrt(np.mean((gt[mask] - pred[mask]) ** 2))

def mae(gt, pred):
    mask = gt > 0
    return np.mean(np.abs(gt[mask] - pred[mask]))

def irmse(gt, pred):
    mask = gt > 0
    return np.sqrt(np.mean((1.0 / gt[mask] - 1.0 / pred[mask]) ** 2))

def imae(gt, pred):
    mask = gt > 0
    return np.mean(np.abs(1.0 / gt[mask] - 1.0 / pred[mask]))

def evaluate(image, gt, sparse, pseudo_depth_map, model, device, args):
    model.eval()

    with torch.no_grad():
        image = torch.FloatTensor(image)
        gt = torch.FloatTensor(gt)
        sparse = torch.FloatTensor(sparse)
        pseudo_depth_map = torch.FloatTensor(pseudo_depth_map)

        if args.cuda:
            image, gt, sparse, pseudo_depth_map = image.cuda(), gt.cuda(), sparse.cuda(), pseudo_depth_map.cuda()
        
        dense_depth = model(image, sparse, pseudo_depth_map, device)

    return dense_depth.cpu().numpy(), gt.cpu().numpy()

def main(rank, world_size, args):
    batch_size = int(args.batch_size / args.gpu_nums)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    setup(rank, world_size, args.master_addr, args.master_port)
    torch.cuda.set_device(rank)

    # 데이터 로드
    val_image, val_sparse, val_gt, val_pseudo_depth_map, val_pseudo_gt_map = lsn.dataloader(args.data_path, mode='val')

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        DA.myImageFloder(val_image, val_sparse, val_gt, val_pseudo_depth_map, val_pseudo_gt_map, True),
        num_replicas=world_size,
        rank=rank
    )

    ValImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(val_image, val_sparse, val_gt, val_pseudo_depth_map, val_pseudo_gt_map, True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=True
    )

    # 모델 로드
    model = DenseLiDAR(batch_size).to(rank)
    model = DDP(model, device_ids=[rank])

    if args.loadmodel is not None:
        checkpoint = torch.load(args.loadmodel, map_location=torch.device('cuda', rank))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Model loaded from {args.loadmodel}')

    torch.cuda.empty_cache()

    total_rmse = 0
    total_mae = 0
    total_irmse = 0
    total_imae = 0
    num_samples = 0

    ## 평가 ##
    print("[Evaluating]")
    for batch_idx, (image, gt, sparse, pseudo_depth_map, pseudo_gt_map) in tqdm(
            enumerate(ValImgLoader), total=len(ValImgLoader), desc="Evaluating"):
        
        pred_depth, ground_truth = evaluate(image, gt, sparse, pseudo_depth_map, model, rank, args)

        for i in range(pred_depth.shape[0]):
            gt_i = ground_truth[i, 0, :, :]
            pred_i = pred_depth[i, 0, :, :]

            total_rmse += rmse(gt_i, pred_i)
            total_mae += mae(gt_i, pred_i)
            total_irmse += irmse(gt_i, pred_i)
            total_imae += imae(gt_i, pred_i)

            num_samples += 1

    # 평균 평가 지표 계산
    avg_rmse = total_rmse / num_samples
    avg_mae = total_mae / num_samples
    avg_irmse = total_irmse / num_samples
    avg_imae = total_imae / num_samples

    print(f'Average RMSE: {avg_rmse:.4f}')
    print(f'Average MAE: {avg_mae:.4f}')
    print(f'Average iRMSE: {avg_irmse:.4f}')
    print(f'Average iMAE: {avg_imae:.4f}')

    cleanup()

if __name__ == '__main__':
    world_size = args.gpu_nums
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
