import torch
import torch.nn as nn
import torch.nn.functional as F

def index_points(points, idx):
    
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def square_distance(src, dst):
    
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        
        xyz = xyz.contiguous()
        points = points.contiguous() if points is not None else None

        if self.group_all:
            new_xyz = torch.zeros(xyz.shape[0], 1, 3).to(xyz.device)
            
            if points is not None:
                points_with_xyz = torch.cat([xyz, points], dim=-1)
                grouped_points = points_with_xyz.permute(0, 2, 1).unsqueeze(-1)
            else:
                grouped_points = xyz.permute(0, 2, 1).unsqueeze(-1)
        else:
            new_xyz_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, new_xyz_idx)
            ball_query_idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, ball_query_idx)
            grouped_xyz -= new_xyz.view(xyz.shape[0], self.npoint, 1, 3)

            if points is not None:
                grouped_points_cat = index_points(points, ball_query_idx)
                grouped_points_cat = torch.cat([grouped_xyz, grouped_points_cat], dim=-1)
            else:
                grouped_points_cat = grouped_xyz
            
            grouped_points = grouped_points_cat.permute(0, 3, 2, 1)
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))

        if self.group_all:
             new_points = F.adaptive_max_pool2d(grouped_points, 1).squeeze(-1)
             new_points = new_points.permute(0, 2, 1)
        else:
            new_points = torch.max(grouped_points, 2)[0]
            new_points = new_points.permute(0, 2, 1)

        return new_xyz, new_points


class PointNetPlusPlusEncoder(nn.Module):
    def __init__(self, output_dim, input_feature_dim=0):
        super(PointNetPlusPlusEncoder, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=input_feature_dim + 3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.proj = nn.Linear(1024, output_dim)
        self.bn_proj = nn.BatchNorm1d(output_dim)


    def forward(self, xyz, features=None):
        
        B, _, _ = xyz.shape
        if features is not None:
             points = torch.cat([xyz, features], dim=2)
        else:
             points = None 
            
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        
        global_feature = l3_points.view(B, 1024)
        global_feature = F.relu(self.bn_proj(self.proj(global_feature)))
        
        return global_feature 