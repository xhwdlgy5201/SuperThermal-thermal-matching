import time
import torch
from utils.math_utils import distance_matrix_vector, pairwise_distances, ptCltoCr

def gct(f="l"):
    """
    get current time
    :param f: "l" for log, "f" for file name
    :return: formatted time
    """
    if f == "l":
        return time.strftime("%m-%d %H:%M:%S", time.localtime(time.time()))
    elif f == "f":
        return f'{time.strftime("%m_%d_%H_%M", time.localtime(time.time()))}'



def nearest_neighbor_match_score(des1, des2, kp1w, kp2, visible, COO_THRSH):
    des_dist_matrix = distance_matrix_vector(des1, des2)
    nn_value, nn_idx = des_dist_matrix.min(dim=-1)

    nn_kp2 = kp2.index_select(dim=0, index=nn_idx)

    coo_dist_matrix = pairwise_distances(
        kp1w[:, 1:3].float(), nn_kp2[:, 1:3].float()
    ).diag()
    correct_match_label = coo_dist_matrix.le(COO_THRSH) * visible

    correct_matches = correct_match_label.sum().item()
    predict_matches = max(visible.sum().item(), 1)

    return correct_matches, predict_matches


def nearest_neighbor_threshold_match_score(
    des1, des2, kp1w, kp2, visible, DES_THRSH, COO_THRSH
):
    des_dist_matrix = distance_matrix_vector(des1, des2)
    nn_value, nn_idx = des_dist_matrix.min(dim=-1)
    predict_label = nn_value.lt(DES_THRSH) * visible

    nn_kp2 = kp2.index_select(dim=0, index=nn_idx)

    coo_dist_matrix = pairwise_distances(
        kp1w[:, 1:3].float(), nn_kp2[:, 1:3].float()
    ).diag()
    correspondences_label = coo_dist_matrix.le(COO_THRSH) * visible

    correct_match_label = predict_label * correspondences_label

    correct_matches = correct_match_label.sum().item()
    predict_matches = max(predict_label.sum().item(), 1)

    return correct_matches, predict_matches


def threshold_match_score(des1, des2, kp1w, kp2, visible, DES_THRSH, COO_THRSH):
    des_dist_matrix = distance_matrix_vector(des1, des2)
    visible = visible.unsqueeze(-1).repeat(1, des_dist_matrix.size(1))
    predict_label = des_dist_matrix.lt(DES_THRSH) * visible

    coo_dist_matrix = pairwise_distances(kp1w[:, 1:3].float(), kp2[:, 1:3].float())
    correspondences_label = coo_dist_matrix.le(COO_THRSH) * visible

    correct_match_label = predict_label * correspondences_label

    correct_matches = correct_match_label.sum().item()
    predict_matches = max(predict_label.sum().item(), 1)
    correspond_matches = max(correspondences_label.sum().item(), 1)

    return correct_matches, predict_matches, correspond_matches


def nearest_neighbor_distance_ratio_match(des1, des2, kp2, threshold):
    des_dist_matrix = distance_matrix_vector(des1, des2)
    sorted, indices = des_dist_matrix.sort(dim=-1)
    Da, Db, Ia = sorted[:, 0], sorted[:, 1], indices[:, 0]
    DistRatio = Da / Db
    predict_label = DistRatio.lt(threshold)
    nn_kp2 = kp2.index_select(dim=0, index=Ia.view(-1))
    return predict_label, nn_kp2


def nearest_neighbor_distance_ratio_match_score(
    des1, des2, kp1w, kp2, visible, COO_THRSH, threshold=0.7
):
    predict_label, nn_kp2 = nearest_neighbor_distance_ratio_match(
        des1, des2, kp2, threshold
    )

    predict_label = predict_label * visible

    coo_dist_matrix = pairwise_distances(
        kp1w[:, 1:3].float(), nn_kp2[:, 1:3].float()
    ).diag()
    correspondences_label = coo_dist_matrix.le(COO_THRSH) * visible

    correct_match_label = predict_label * correspondences_label

    correct_matches = correct_match_label.sum().item()
    predict_matches = max(predict_label.sum().item(), 1)

    return correct_matches, predict_matches



def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def prettydict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("\t" * indent + f"{key}")
            prettydict(value, indent + 1)
        else:
            print("\t" * indent + f"{key:>18} : {value}")


def unsqueezebatch(batch):
    for key in batch:
        batch[key] = batch[key].unsqueeze(0)
    return batch


def isnan(t):
    return torch.isnan(t).sum().item() > 0


def imgBatchXYZ(B, H, W):
    Ha, Wa = torch.arange(H), torch.arange(W)
    gy, gx = torch.meshgrid([Ha, Wa])
    gx, gy = torch.unsqueeze(gx.float(), -1), torch.unsqueeze(gy.float(), -1)
    ones = gy.new_full(gy.size(), fill_value=1)
    grid = torch.cat((gx, gy, ones), -1)  # (H, W, 3)
    grid = torch.unsqueeze(grid, 0)  # (1, H, W, 3)
    grid = grid.repeat(B, 1, 1, 1)  # (B, H, W, 3)
    return grid


def transXYZ_2_to_1(batchXYZ, homo21):
    """
    project each pixel in right to left xy coordination
    :param batchXYZ: (B, H, W, 3)
    :param homo21: (B, 3, 3)
    :return: warp XYZ: (B, H, W, 2)
    """
    B, H, W, C = batchXYZ.size()
    grid = batchXYZ.contiguous().view(B, H * W, C)  # (B, H*W, 3)
    grid = grid.contiguous().permute(0, 2, 1)  # (B, 3, H*W)
    grid = grid.type_as(homo21).to(homo21.device)

    grid_w = torch.matmul(homo21, grid)  # (B, 3, 3) matmul (B, 3, H*W) => (B, 3, H*W)
    grid_w = grid_w.contiguous().permute(0, 2, 1)  # (B, H*W, 3)
    grid_w = torch.div(
        grid_w, torch.unsqueeze(grid_w[:, :, 2], -1) + 1e-8
    )  # (B, H*W, 3)
    grid_w = grid_w.contiguous().view(B, H, W, -1)[:, :, :, :2]  # (B, H, W, 2)

    return grid_w
