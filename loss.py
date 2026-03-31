import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss, complete_box_iou_loss

def make_grid(feature_map, stride):
    """
    生成特徵圖對應原圖的座標網格 (x, y)
    """
    B, C, H, W = feature_map.shape
    device = feature_map.device
    
    # 產生 0 ~ W-1 和 0 ~ H-1
    xs = torch.arange(0, W, device=device) * stride
    ys = torch.arange(0, H, device=device) * stride
    
    # 生成網格
    y, x = torch.meshgrid(ys, xs, indexing='ij')
    
    # [H*W, 2] -> 代表每個 pixel 在原圖的中心座標
    # 加上 stride // 2 是為了讓點落在格子的中心
    center_points = torch.stack([x, y], dim=-1) + stride // 2
    return center_points

def distance2bbox(points, distance):
    """
    將預測的距離 (l, t, r, b) 轉換為 bounding box (x1, y1, x2, y2)
    Args:
        points: [N, 2] (cx, cy)
        distance: [N, 4] (l, t, r, b)
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return torch.stack([x1, y1, x2, y2], -1)

class FCOSLoss(nn.Module):
    def __init__(self, strides=[8, 16, 32], center_sampling_radius=1.5):
        """
        Args:
            strides: FPN 每層的 Stride
            center_sampling_radius: Center Sampling 的半徑倍率 (預設 1.5)
        """
        super().__init__()
        self.strides = strides
        self.center_sampling_radius = center_sampling_radius
        self.center_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, preds, targets):
        """
        Args:
            preds: (cls_logits, bbox_regression, centerness) 來自模型的輸出
            targets: List of dict, 每個 dict 包含 'boxes' (Tensor[N, 4]) 和 'labels' (Tensor[N])
        """
        cls_logits, bbox_reg, centerness = preds
        
        # 1. 收集所有層的輸出並 Flatten (包含 Strides 資訊)
        # flatten_strides 用於 Center Sampling 計算半徑
        flatten_cls, flatten_reg, flatten_ctr, grids, flatten_strides = self._flatten_inputs(cls_logits, bbox_reg, centerness)
        
        # 2. 製作 Ground Truth Targets (包含 Center Sampling 邏輯)
        target_cls, target_reg, target_ctr = self._prepare_targets(
            grids, targets, flatten_cls.shape[0], flatten_strides
        )
        
        # 建立正樣本遮罩
        pos_mask = target_cls > 0
        num_pos = pos_mask.sum().item()
        num_pos = max(num_pos, 1.0)

        # --- LOSS 1: Classification (Focal Loss) ---
        loss_cls = sigmoid_focal_loss(
            flatten_cls.view(-1), 
            target_cls.float(), 
            alpha=0.25, 
            gamma=2.0, 
            reduction='sum'
        ) / num_pos

        # --- LOSS 2: Regression (GIoU Loss) ---
        loss_reg = torch.tensor(0.0).to(flatten_cls.device)
        if pos_mask.sum() > 0:
            pos_grids = grids[pos_mask]
            pos_bbox_pred = flatten_reg[pos_mask]
            pos_bbox_targets = target_reg[pos_mask]
            
            # 將距離 (l,t,r,b) 轉回 (x1, y1, x2, y2) 才能算 GIoU
            pred_boxes = distance2bbox(pos_grids, pos_bbox_pred)
            target_boxes = distance2bbox(pos_grids, pos_bbox_targets)
            
            loss_reg = complete_box_iou_loss(pred_boxes, target_boxes, reduction='sum') / num_pos

        # --- LOSS 3: Centerness (BCE Loss) ---
        loss_ctr = torch.tensor(0.0).to(flatten_cls.device)
        if pos_mask.sum() > 0:
            pos_ctr_pred = flatten_ctr[pos_mask].view(-1)
            pos_ctr_targets = target_ctr[pos_mask]
            loss_ctr = self.center_loss_fn(pos_ctr_pred, pos_ctr_targets) / num_pos

        return loss_cls, loss_reg, loss_ctr

    def _flatten_inputs(self, cls_logits, bbox_reg, centerness):
        """
        修正後的 Flatten 邏輯：確保同一張圖片的所有 FPN 層數據是連續的。
        結構應為：[Batch, Sum(H*W), C] -> Flatten -> [Batch*Sum(H*W), C]
        新增: 回傳 strides 以便計算 Center Sampling 半徑
        """
        all_cls = []
        all_reg = []
        all_ctr = []
        all_grids = []
        all_strides = [] # [新增]
        
        for i, stride in enumerate(self.strides):
            batch_size = cls_logits[i].shape[0]
            h, w = cls_logits[i].shape[-2:]
            
            # 生成 Grid: [H, W, 2] -> [H*W, 2]
            grid = make_grid(cls_logits[i], stride).view(-1, 2) 
            grid = grid.unsqueeze(0).expand(batch_size, -1, -1)
            
            # [新增] 生成 Stride Tensor: [Batch, H*W]
            stride_tensor = torch.full((batch_size, h * w), stride, device=cls_logits[i].device)

            # 處理預測值: [B, C, H, W] -> [B, H*W, C]
            c = cls_logits[i].flatten(2).permute(0, 2, 1)
            r = bbox_reg[i].flatten(2).permute(0, 2, 1)
            ctr = centerness[i].flatten(2).permute(0, 2, 1)
            
            all_cls.append(c)
            all_reg.append(r)
            all_ctr.append(ctr)
            all_grids.append(grid)
            all_strides.append(stride_tensor)
            
        # 在維度 1 拼接
        flatten_cls = torch.cat(all_cls, dim=1)
        flatten_reg = torch.cat(all_reg, dim=1)
        flatten_ctr = torch.cat(all_ctr, dim=1)
        flatten_grids = torch.cat(all_grids, dim=1)
        flatten_strides = torch.cat(all_strides, dim=1) # [Batch, Total_Points]
        
        return (flatten_cls.reshape(-1, 1), 
                flatten_reg.reshape(-1, 4), 
                flatten_ctr.reshape(-1, 1), 
                flatten_grids.reshape(-1, 2),
                flatten_strides.reshape(-1))

    @torch.no_grad()
    def _prepare_targets(self, grids, targets, total_num_points, strides):
        """
        全矩陣運算版本的 Target Assignment (加入 Center Sampling)。
        strides: [Batch * N_points] -> 需要 reshape 成 [Batch, N_points]
        """
        batch_size = len(targets)
        num_points_per_img = total_num_points // batch_size
        
        # 1. 還原維度
        grids = grids.view(batch_size, num_points_per_img, 2)
        strides = strides.view(batch_size, num_points_per_img)

        # 2. 處理 Ground Truth (Padding)
        gt_boxes_list = [t['boxes'] for t in targets]
        gt_labels_list = [t['labels'] for t in targets]
        
        max_num_gt = max([len(b) for b in gt_boxes_list])
        
        if max_num_gt == 0:
            return (torch.zeros(total_num_points, device=grids.device),
                    torch.zeros((total_num_points, 4), device=grids.device),
                    torch.zeros(total_num_points, device=grids.device))

        gt_boxes_padded = torch.zeros((batch_size, max_num_gt, 4), dtype=torch.float32, device=grids.device)
        gt_labels_padded = torch.zeros((batch_size, max_num_gt), dtype=torch.float32, device=grids.device)
        gt_valid_mask = torch.zeros((batch_size, max_num_gt), dtype=torch.bool, device=grids.device)

        for i, (boxes, labels) in enumerate(zip(gt_boxes_list, gt_labels_list)):
            num_gt = len(boxes)
            if num_gt > 0:
                gt_boxes_padded[i, :num_gt] = boxes
                gt_labels_padded[i, :num_gt] = labels
                gt_valid_mask[i, :num_gt] = True

        # 3. 計算點到框的距離 (Original Box)
        # grids: [Batch, N_points, 1, 2]
        # gt_boxes: [Batch, 1, Max_GT, 4]
        
        lt = grids[:, :, None, :] - gt_boxes_padded[:, None, :, :2] 
        rb = gt_boxes_padded[:, None, :, 2:] - grids[:, :, None, :]
        
        reg_targets_all = torch.cat([lt, rb], dim=3) # [Batch, N_points, Max_GT, 4]
        min_dist_box, _ = reg_targets_all.min(dim=-1)
        is_in_box = min_dist_box > 0 # 條件 A: 在原始框內

        # --- [新增] Center Sampling 判定 ---
        # 4. 計算點到 "Center Box" 的距離
        # Center Box 定義: 以 GT 中心為圓心，半徑為 radius * stride
        
        gt_cx = (gt_boxes_padded[..., 0] + gt_boxes_padded[..., 2]) / 2 # [Batch, Max_GT]
        gt_cy = (gt_boxes_padded[..., 1] + gt_boxes_padded[..., 3]) / 2
        
        # 計算點與 GT 中心的位移 (x-cx, y-cy)
        # grids: [Batch, N_points, 1, 2] - gt_centers: [Batch, 1, Max_GT, 2] (手動堆疊)
        dx = grids[:, :, None, 0] - gt_cx[:, None, :] # [Batch, N_points, Max_GT]
        dy = grids[:, :, None, 1] - gt_cy[:, None, :] 

        # 計算該點允許的半徑 (stride * sampling_ratio)
        # strides: [Batch, N_points] -> [Batch, N_points, 1] 以廣播到所有 GT
        radii = strides[:, :, None] * self.center_sampling_radius
        
        # 條件 B: 在 Center Box 內 (絕對距離 < 半徑)
        is_in_center = (dx.abs() < radii) & (dy.abs() < radii)

        # 5. 綜合判定 (Valid Match)
        # 必須同時滿足：(1) 在原始框內 (2) 在 Center Box 內 (3) 是有效的 GT
        is_valid_gt = gt_valid_mask[:, None, :].expand_as(is_in_box)
        valid_match = is_in_box & is_in_center & is_valid_gt

        # 6. 處理 Ambiguity (選面積最小的)
        gt_areas = (gt_boxes_padded[..., 2] - gt_boxes_padded[..., 0]) * \
                   (gt_boxes_padded[..., 3] - gt_boxes_padded[..., 1])
                   
        gt_areas_expanded = gt_areas[:, None, :].repeat(1, num_points_per_img, 1)
        INF = 100000000.0
        gt_areas_expanded[~valid_match] = INF # 不合法的設為無限大
        
        min_area, min_area_idx = gt_areas_expanded.min(dim=2) # [Batch, N_points]
        
        pos_mask = min_area != INF

        # 7. 收集 Targets (Gather)
        # Labels
        gt_labels_expanded = gt_labels_padded[:, None, :].expand(batch_size, num_points_per_img, max_num_gt)
        target_cls = gt_labels_expanded.gather(2, min_area_idx.unsqueeze(-1)).squeeze(-1)
        target_cls[~pos_mask] = 0.0

        # Regression
        min_area_idx_expanded = min_area_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 4)
        target_reg = reg_targets_all.gather(2, min_area_idx_expanded).squeeze(2)
        target_reg[~pos_mask] = 0.0

        # Centerness
        l_r = target_reg[:, :, 0::2]
        t_b = target_reg[:, :, 1::2]
        ctr_lr = l_r.min(dim=-1)[0] / (l_r.max(dim=-1)[0] + 1e-6)
        ctr_tb = t_b.min(dim=-1)[0] / (t_b.max(dim=-1)[0] + 1e-6)
        target_ctr = torch.sqrt(ctr_lr * ctr_tb)
        target_ctr[~pos_mask] = 0.0

        return (target_cls.view(-1), 
                target_reg.view(-1, 4), 
                target_ctr.view(-1))