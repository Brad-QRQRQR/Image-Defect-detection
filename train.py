import os
import sys
import shutil

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models

from torchvision.ops import nms
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from torchvision.transforms import v2

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from dataset import *
from utils import *
from loss import FCOSLoss, distance2bbox
from model_fcos import CorruptionDetector

def compute_iou_elementwise(pred_boxes, target_boxes):
    """
    計算成對框的 IoU
    Args:
        pred_boxes: [N, 4] (x1, y1, x2, y2)
        target_boxes: [N, 4] (x1, y1, x2, y2)
    Returns:
        iou: [N]
    """
    # 1. 計算個別面積
    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    area_target = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0) * \
                  (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=0)

    # 2. 計算交集 (Intersection) 座標
    lt = torch.max(pred_boxes[:, :2], target_boxes[:, :2]) # [N, 2]
    rb = torch.min(pred_boxes[:, 2:], target_boxes[:, 2:]) # [N, 2]

    # 3. 計算交集面積
    wh = (rb - lt).clamp(min=0) # [N, 2]
    inter = wh[:, 0] * wh[:, 1]

    # 4. 計算聯集 (Union)
    union = area_pred + area_target - inter

    # 5. IoU
    iou = inter / (union + 1e-6)
    return iou

@torch.no_grad()
def calc_statistics(outputs, targets, criterion, conf_threshold=0.5):
    """
    [優化版] 計算當前 Batch 的統計數據 (全矩陣運算，無 Python Loop)
    """
    cls_logits, bbox_reg, centerness = outputs
    device = cls_logits[0].device
    batch_size = cls_logits[0].shape[0]

    # --- 1. 快速計算 Image-Level Accuracy ---
    
    # A. 處理 Ground Truth (GT)
    # 只要 boxes 數量 > 0 就是 Corrupted (1)，否則為 Normal (0)
    # 列表生成式在 Python 中處理小量數據(如 Batch=64)非常快
    gt_labels = torch.tensor([t['labels'] > 0 for t in targets], 
                             dtype=torch.float32, device=device)

    max_scores_per_level = []
    
    # 這裡只會跑 3 次 (對應 FPN P3, P4, P5)，開銷極小
    for i in range(len(cls_logits)):
        # 1. 結合分數: [Batch, 1, H, W]
        #    利用 sigmoid() 把數值壓到 0~1
        level_score = (cls_logits[i].sigmoid() * centerness[i].sigmoid()).sqrt()
        
        # 2. 展平並取該層最大值: [Batch, H*W] -> max(dim=1) -> [Batch]
        #    這步 GPU 會同時幫 64 張圖找最大值，不用寫迴圈
        level_max = level_score.view(batch_size, -1).max(dim=1)[0]
        
        max_scores_per_level.append(level_max)
    
    # 3. 跨層級取最大值: [3, Batch] -> max(dim=0) -> [Batch]
    #    這就是每張圖片最終的「最高信心分數」
    final_scores = torch.stack(max_scores_per_level, dim=0).max(dim=0)[0]
    
    # 4. 判定分類 & 計算答對數量
    pred_labels = (final_scores > conf_threshold).float()
    correct_count = (pred_labels == gt_labels).float().sum()

    # --- 2. 計算 Mean IoU (維持不變) ---
    # IoU 計算依賴複雜的 Target Assignment，這部分已經是矩陣操作，無需大幅改動
    # 除非改寫 Loss 讓它回傳 assignment 結果，否則這裡必須重新計算一次 assignment
    
    iou_sum = 0.0
    num_pos_samples = 0
    
    # 只有當存在正樣本時才計算 IoU (避免 unnecessary computation)
    # 這裡可以做一個快速檢查：如果 GT 全是背景，直接回傳 0
    if gt_labels.sum() > 0:
        flatten_cls, flatten_reg, _, grids, flatten_strides = criterion._flatten_inputs(cls_logits, bbox_reg, centerness)
        target_cls, target_reg, _ = criterion._prepare_targets(grids, targets, flatten_cls.shape[0], flatten_strides)
        
        pos_mask = target_cls > 0
        num_pos_samples = pos_mask.sum().item()
        if pos_mask.any():
            pos_pred_reg = flatten_reg[pos_mask]
            pos_target_reg = target_reg[pos_mask]
            pos_grids = grids[pos_mask]
            
            # 假設 distance2bbox 和 compute_iou_elementwise 已經 import
            pred_boxes = distance2bbox(pos_grids, pos_pred_reg)
            target_boxes = distance2bbox(pos_grids, pos_target_reg)
            
            ious = compute_iou_elementwise(pred_boxes, target_boxes)
            iou_sum = ious.sum()

    return correct_count, iou_sum, num_pos_samples

def post_process_fcos(outputs, strides, image_size, conf_thresh=0.05, nms_thresh=0.6):
    """
    將 FCOS 的 raw outputs 轉換為標準的 bbox 列表
    Args:
        outputs: (cls_logits, bbox_reg, centerness)
        strides: [8, 16, 32] (對應 P3, P4, P5)
        image_size: (H, W) 原始圖片大小，用來切除超出邊界的框
    Returns:
        results: List[Dict], 每個元素對應一張圖片 {'boxes':..., 'scores':..., 'labels':...}
    """
    cls_logits, bbox_reg, centerness = outputs
    batch_size = cls_logits[0].shape[0]
    device = cls_logits[0].device
    
    results = []

    # 針對 Batch 中的每一張圖片個別處理
    for img_idx in range(batch_size):
        img_boxes = []
        img_scores = []
        img_labels = []

        # 遍歷每一層 (P3, P4, P5)
        for i, stride in enumerate(strides):
            # 1. 取得該層的輸出並移除 Batch 維度 -> [C, H, W]
            cls_score = cls_logits[i][img_idx].sigmoid()
            ctr_score = centerness[i][img_idx].sigmoid()
            reg_pred = bbox_reg[i][img_idx] # [4, H, W]

            H, W = cls_score.shape[-2:]

            # 2. 生成網格 (Grid Coordinates)
            shift_x = torch.arange(0, W, device=device) * stride + stride // 2
            shift_y = torch.arange(0, H, device=device) * stride + stride // 2
            gy, gx = torch.meshgrid(shift_y, shift_x, indexing='ij')
            grids = torch.stack((gx, gy), dim=-1) # [H, W, 2]

            # 3. 計算綜合信心分數 (Cls * Ctr)
            # sqrt 是一個常用技巧，讓分數不要因為兩個小數相乘掉太快
            final_scores = (cls_score * ctr_score).sqrt() # [C, H, W]
            
            # [Fix] 關鍵修正：如果是單類別 (C=1)，移除 Channel 維度 [1, H, W] -> [H, W]
            # 這樣生成的 mask 才會是 [H, W]，才能正確索引 [H, W, 4] 的 reg_pred
            if final_scores.dim() == 3 and final_scores.shape[0] == 1:
                final_scores = final_scores.squeeze(0)

            # 4. 閾值過濾 (加速 NMS)
            keep_mask = final_scores > conf_thresh
            if not keep_mask.any():
                continue
            
            # 篩選出通過閾值的點
            score_mask = final_scores[keep_mask]
            
            # 這裡假設你是單類別 (Normal vs Corrupted)
            label_mask = torch.ones_like(score_mask, dtype=torch.int64) 

            # 5. 解碼 Bounding Box
            # reg_pred: [4, H, W] -> [H, W, 4]
            # 現在 keep_mask 是 [H, W]，可以正確索引 [H, W, 4] 的 tensor
            reg_mask = reg_pred.permute(1, 2, 0)[keep_mask] # [N_keep, 4]
            grid_mask = grids[keep_mask]                    # [N_keep, 2]

            # 根據 FCOS 公式：box = grid +/- reg
            # 假設 reg 順序是 (l, t, r, b)
            x1 = grid_mask[:, 0] - reg_mask[:, 0]
            y1 = grid_mask[:, 1] - reg_mask[:, 1]
            x2 = grid_mask[:, 0] + reg_mask[:, 2]
            y2 = grid_mask[:, 1] + reg_mask[:, 3]
            
            # Clip boxes to image size
            x1 = x1.clamp(min=0, max=image_size[1])
            y1 = y1.clamp(min=0, max=image_size[0])
            x2 = x2.clamp(min=0, max=image_size[1])
            y2 = y2.clamp(min=0, max=image_size[0])

            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            img_boxes.append(boxes)
            img_scores.append(score_mask)
            img_labels.append(label_mask)

        # 6. 堆疊所有層級的結果
        if len(img_boxes) > 0:
            img_boxes = torch.cat(img_boxes, dim=0)
            img_scores = torch.cat(img_scores, dim=0)
            img_labels = torch.cat(img_labels, dim=0)

            # 7. 執行 NMS
            keep_indices = nms(img_boxes, img_scores, nms_thresh)
            
            results.append({
                'boxes': img_boxes[keep_indices],
                'scores': img_scores[keep_indices],
                'labels': img_labels[keep_indices]
            })
        else:
            # 該圖片沒有任何檢測結果
            results.append({
                'boxes': torch.zeros((0, 4), device=device),
                'scores': torch.zeros((0,), device=device),
                'labels': torch.zeros((0,), dtype=torch.int64, device=device)
            })
            
    return results

# --- 1. 定義 Collate Function ---
# 這是物件偵測訓練最容易卡關的地方。
# 預設的 DataLoader 會嘗試將 list of dicts 疊加成一個 tensor，這會失敗。
def fcos_collate_fn(batch):
    """
    batch: List of tuples (image, target)
    """
    images, targets = zip(*batch)
    # 將圖片堆疊成 [B, C, H, W]
    images = torch.stack(images, dim=0)
    # Targets 保持為 List[Dict]
    return images, list(targets)

# --- 2. Train One Epoch ---
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    
    tot = 0
    pos_tot = 0
    logger = {
        'loss': torch.zeros(1, device=device),
        'loss_cls': torch.zeros(1, device=device),
        'loss_reg': torch.zeros(1, device=device),
        'loss_ctr': torch.zeros(1, device=device),
        'acc': torch.zeros(1, device=device),
        'iou': torch.zeros(1, device=device)
    }
    
    print("Start Training!!!")
    for images, targets in tqdm(loader):
        images = images.to(device)
        # targets 內的 tensor 也要搬到 device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        # 使用混合精度訓練
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images)
            # outputs = (cls_logits, bbox_reg, centerness)
            
            loss_cls, loss_reg, loss_ctr = criterion(outputs, targets)
            loss_reg *= 5.0
            total_loss = loss_cls + loss_reg + loss_ctr

        # Backward (使用 Scaler)
        scaler.scale(total_loss).backward()
        
        # 梯度裁切 (可選，防止梯度爆炸，對 FPN/ResNet 很有效)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        scaler.step(optimizer)
        scaler.update()

        # --- 新增: 計算統計數據 ---
        batch_acc, batch_iou, pos_num = calc_statistics(outputs, targets, criterion)

        # 累加紀錄
        logger['loss'] += total_loss.detach()
        logger['loss_cls'] += loss_cls.detach()
        logger['loss_reg'] += loss_reg.detach()
        logger['loss_ctr'] += loss_ctr.detach()
        logger['acc'] += batch_acc
        logger['iou'] += batch_iou
        tot += images.size(0)
        pos_tot += pos_num
    
    metric = {k: v.item() / len(loader) for k, v in logger.items() if k not in ['iou', 'acc']}
    metric['iou'] = logger['iou'].item() / pos_tot if pos_tot > 0 else 0
    metric['acc'] = logger['acc'].item() / tot

    return metric

# --- 3. Validate One Epoch ---
@torch.no_grad()
def val_one_epoch(model, loader, criterion, device):
    model.eval()

    map_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=False).to(device)
    strides = [8, 16, 32]

    tot = 0
    pos_tot = 0
    logger = {
        'loss': torch.zeros(1, device=device),
        'loss_cls': torch.zeros(1, device=device),
        'loss_reg': torch.zeros(1, device=device),
        'loss_ctr': torch.zeros(1, device=device),
        'acc': torch.zeros(1, device=device),
        'iou': torch.zeros(1, device=device)
    }
    
    print("Start Validating!!!")
    for images, targets in tqdm(loader):
        images = images.to(device)
        targets = [{k: v.to(device) if k != 'labels' else v.to(device, dtype=torch.int64) 
                    for k, v in t.items()} for t in targets]

        # 驗證時不需要 autocast，但使用也無妨，這裡保持簡單
        outputs = model(images)
        loss_cls, loss_reg, loss_ctr = criterion(outputs, targets)
        loss_reg *= 5.0
        total_loss = loss_cls + loss_reg + loss_ctr
        

        # --- 新增: 計算統計數據 ---
        batch_acc, batch_iou, pos_num = calc_statistics(outputs, targets, criterion)

        # ==========================================
        # 新增: 收集預測結果 (Preds)
        # ==========================================
        # 1. 將 Raw Output 轉為 Bbox 列表
        # 注意: image_size 假設是 (320, 320)，從 dataset 設定來的
        preds = post_process_fcos(outputs, strides, image_size=(IMG_SIZE[1], IMG_SIZE[0]))
        # 2. 更新 Metric
        # targets 格式剛好符合: [{'boxes':..., 'labels':...}]
        map_metric.update(preds, targets)
        # ==========================================

        # 累加紀錄
        logger['loss'] += total_loss.detach()
        logger['loss_cls'] += loss_cls.detach()
        logger['loss_reg'] += loss_reg.detach()
        logger['loss_ctr'] += loss_ctr.detach()
        logger['acc'] += batch_acc
        logger['iou'] += batch_iou
        tot += images.size(0)
        pos_tot += pos_num

    # --- 迴圈結束後計算 mAP ---
    map_result = map_metric.compute()

    metric = {k: v.item() / len(loader) for k, v in logger.items() if k not in ['iou', 'acc']}
    metric['iou'] = logger['iou'].item() / pos_tot if pos_tot > 0 else 0
    metric['acc'] = logger['acc'].item() / tot
    metric['map'] = map_result['map'].item()
    metric['map_50'] = map_result['map_50'].item()

    return metric

def train(model, train_dataset, test_dataset, history, device, log_file, batch_size=32, epochs=50, lr=1e-4):
    best_map = 0
    best_map_50 = 0

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3, fused=True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5,
        min_lr=1e-6,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=get_sampler(get_all_labels(train_dataset), 42),
        shuffle=False,
        collate_fn=fcos_collate_fn,
        num_workers=3,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=fcos_collate_fn,
        num_workers=2,
        pin_memory=True,         # [新增] 加速 CPU -> GPU
        persistent_workers=True, # [新增] 避免每個 Epoch 重啟驗證進程
    )

    scaler = GradScaler('cuda') # 負責管理梯度縮放

    criterion = FCOSLoss(strides=[8, 16, 32])

    early_stopping = EarlyStopping(
        patience=8,
        delta=0.001,
        verbose=True
    )

    for epoch in range(epochs):
        # --- Training Phase ---
        t_log = train_one_epoch(
            model, train_dataloader, optimizer, criterion, scaler, device
        )
        
        # --- Validation Phase ---
        v_log = val_one_epoch(
            model, test_dataloader, criterion, device
        )

        scheduler.step(v_log['loss'])
        
        # --- Update History ---
        history['train_acc'].append(t_log['acc'])
        history['train_iou'].append(t_log['iou'])
        history['train_loss'].append(t_log['loss'])
        history['train_cls_loss'].append(t_log['loss_cls'])
        history['train_reg_loss'].append(t_log['loss_reg'])
        history['train_ctr_loss'].append(t_log['loss_ctr'])

        history['val_acc'].append(v_log['acc'])
        history['val_iou'].append(v_log['iou'])
        history['val_loss'].append(v_log['loss'])
        history['val_cls_loss'].append(v_log['loss_cls'])
        history['val_reg_loss'].append(v_log['loss_reg'])
        history['val_ctr_loss'].append(v_log['loss_ctr'])
        history['val_map'].append(v_log['map'])
        history['val_map_50'].append(v_log['map_50'])

        history['lr'].append(optimizer.param_groups[0]['lr']) # 紀錄 LR 變化
        
        # --- Logging ---
        log_msg = (
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {t_log['loss']:.4f} (Cls: {t_log['loss_cls']:.4f}, Reg: {t_log['loss_reg']:.4f}, Ctr: {t_log['loss_ctr']:.4f}) "
            f"Train Acc: {t_log['acc']:.4f} Train IoU: {t_log['iou']:.4f} | "
            f"Val Loss: {v_log['loss']:.4f} (Cls: {v_log['loss_cls']:.4f}, Reg: {v_log['loss_reg']:.4f}, Ctr: {v_log['loss_ctr']:.4f}) "
            f"Val Acc: {v_log['acc']:.4f} Val IoU: {v_log['iou']:.4f} "
            f"Val Map: {v_log['map']:.4f} Val Map_50: {v_log['map_50']:.4f} | "
            f"lr: {history['lr'][-1]}"
        )
        
        print(log_msg)
        log_file.write(log_msg + "\n")

        if v_log['map'] > best_map:
            best_map = v_log['map']
            torch.save(model.state_dict(), f"./{TEST_FOLDER}/best_model_map.pth")
        if v_log['map_50'] > best_map_50:
            best_map_50 = v_log['map_50']
            torch.save(model.state_dict(), f"./{TEST_FOLDER}/best_model_map_50.pth")

        early_stopping(v_log['loss'])
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    torch.save(model.state_dict(), f"./{TEST_FOLDER}/model.pth")

    return best_map, best_map_50

# 放在 model = CorruptionDetector().to(device) 之後
def freeze_backbone(model):
    print("Freezing backbone layers (Stem)...")
    for param in model.backbone_fpn.stem.parameters():
        param.requires_grad = False

def prepare():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CorruptionDetector().to(device)
    freeze_backbone(model)
    # model = torch.compile(model)
    history = {
        'train_loss': [],
        'train_cls_loss': [],
        'train_reg_loss': [],
        'train_ctr_loss': [],
        'train_acc': [],
        'train_iou': [],
        'val_loss': [],
        'val_cls_loss': [],
        'val_reg_loss': [],
        'val_ctr_loss': [],
        'val_acc': [],
        'val_iou': [],
        'val_map': [],
        'val_map_50': [],
        'lr': [],
    }
    return model, history, device

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    SEED = 42
    TEST_FOLDER = "fcol_320by320_final10"
    IMG_SIZE = (512, 288)

    if not os.path.exists(TEST_FOLDER):
        os.mkdir(f"./{TEST_FOLDER}")
    else:
        print("existed!!! (press q to exit)")
        action = input()
        if action == 'q':
            sys.exit(0)
        else:
            shutil.rmtree(TEST_FOLDER)
            os.mkdir(f"./{TEST_FOLDER}")
    
    saver(TEST_FOLDER)
    log_file = open(f"./{TEST_FOLDER}/record.txt", "w", encoding="utf-8")
    model, history, device = prepare()

    
    train_dataset = CorruptionLMDBDataset(
        lmdb_path=r"D:\comVis\content\train_lmdb",
        transform=v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomAffine(
                degrees=10, 
                translate=(0.1, 0.1), 
                scale=(0.8, 1.2) 
            ),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        target_size=IMG_SIZE,
    )
    test_dataset = CorruptionLMDBDataset(
        lmdb_path=r"D:\comVis\content\test_lmdb",
        transform=v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        target_size=IMG_SIZE,
    )
    
    best_map, best_map_50 = train(model, train_dataset, test_dataset, history, device, log_file)

    show_train_history(history, 'train_loss', 'val_loss', 'Loss', f"./{TEST_FOLDER}/loss.png")
    show_train_history(history, 'train_cls_loss', 'val_cls_loss', 'Cls loss', f"./{TEST_FOLDER}/loss_cls.png")
    show_train_history(history, 'train_reg_loss', 'val_reg_loss', 'Reg loss', f"./{TEST_FOLDER}/loss_reg.png")
    show_train_history(history, 'train_ctr_loss', 'val_ctr_loss', 'Ctr loss', f"./{TEST_FOLDER}/loss_ctr.png")
    show_train_history(history, 'train_acc', 'val_acc', 'Acc', f"./{TEST_FOLDER}/accuracy.png")
    show_train_history(history, 'train_iou', 'val_iou', 'Iou', f"./{TEST_FOLDER}/iou.png")
    show_train_history(history,  None, 'val_map', 'map', f"./{TEST_FOLDER}/map.png")
    show_train_history(history,  None, 'val_map_50', 'map_50', f"./{TEST_FOLDER}/map_50.png")

    log_file.write(f"best map: {best_map} | best map_50: {best_map_50}")