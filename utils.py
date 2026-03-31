import os
import torch
import random
import numpy as np
from tqdm import tqdm
import pickle

import time
from shutil import copyfile

def get_all_labels(dataset):
    """
    [極速版] 直接讀取 LMDB 的 Bytes，跳過 Image 解碼與 Transform。
    """
    labels = []
    print("正在掃描 Dataset 以建立採樣權重 (Fast Mode)...")
    
    # 確保 LMDB 環境已初始化
    if dataset.env is None:
        dataset._init_db()
    
    # 使用 LMDB 的 transaction 進行讀取
    with dataset.env.begin(write=False) as txn:
        # 你的 dataset key 是 0, 1, 2... 的 ASCII 字串
        for i in tqdm(range(len(dataset))):
            key = str(i).encode('ascii')
            byteflow = txn.get(key)
            
            if byteflow is None:
                continue
                
            # 這裡只做反序列化 (Unpickle)，不解碼圖片
            # 雖然 pickle.loads 會把圖片 bytes 也讀進記憶體，
            # 但比起 Image.open + Resize + Normalize，這快了 100 倍以上
            sample = pickle.loads(byteflow)
            labels.append(sample['label'])
            
    return labels

def get_sampler(labels, seed):
    """
    計算 WeightedRandomSampler
    """
    labels = np.array(labels)
    class_counts = np.bincount(labels) # 計算 [正常數量, 損壞數量]
    
    # 計算每個類別的權重 (數量越少，權重越大)
    # 權重 = 1 / 數量
    class_weights = 1. / class_counts
    
    # 為"每一個樣本"分配權重
    # 如果樣本 i 是 label 1，它的權重就是 class_weights[1]
    samples_weights = class_weights[labels]

    # with open(f"./weights/wei.txt", "w", encoding="utf-8") as f:
    #     f.write(f"labels: {labels}\n")
    #     f.write(f"class: {class_counts}\n")
    #     f.write(f"class weights: {class_weights}\n")
    #     f.write(f"weights: {samples_weights}\n")

    g = torch.Generator()
    g.manual_seed(seed)
    
    # 建立 Sampler
    # replacement=True 代表可以重複抽樣 (這是 Oversampling 的關鍵)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.DoubleTensor(samples_weights),
        num_samples=len(samples_weights),
        replacement=True,
        generator=g
    )
    return sampler


def seed_everything(seed=42):
    """
    鎖定所有隨機生成的種子，確保實驗可重現。
    """
    # 1. Python 原生隨機
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止 hash 隨機化

    # 2. NumPy 隨機 (Albumentations 主要依賴這個)
    np.random.seed(seed)

    # 3. PyTorch 隨機 (CPU & GPU) (Torchvision transform 依賴這個)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print(f'Global seed set to {seed}')

def worker_init_fn(worker_id):
    # 每個 worker 的 seed 必須不同，否則所有 worker 會做出一模一樣的增強
    # 我們用全域 seed + worker_id 來區分
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def saver(test_dir):
    localtime = time.localtime()
    safe_time = time.strftime("%Y-%m-%d_%H-%M-%S", localtime)
    f = open(os.path.join(test_dir, safe_time), "w+", encoding="utf-8")
    f.close()

    src_folder = './'

    file_list = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    for file_name in file_list:
        src = os.path.join(src_folder, file_name)
        dst = os.path.join(test_dir, file_name)
        copyfile(src, dst)

import matplotlib.pyplot as plt
def show_train_history(train_history, train, validation, label, save_path=None):
    if not train and not validation:
        print(f"both category are None.")
        return
    if train:
        plt.plot(train_history[train])
    if validation:
        plt.plot(train_history[validation])
    plt.title('Train History')
    plt.ylabel(label)
    plt.xlabel('Epoch')
    if train and validation:
        plt.legend(['train','validation'],loc='upper left')
    elif train:
        plt.legend(['train'],loc='upper left')
    else:
        plt.legend(['validation'],loc='upper left')
    if save_path:
        plt.savefig(save_path)
    plt.show()


import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
# 不需要 from torchvision.ops import nms，因為 post_process_fcos 已經做了

def visualize_debug(model, dataset, device, idx=0, conf_thresh=0.01):
    """
    診斷視覺化核心函數 (增強版：加入 Image Confidence 計算與最高分框標示)
    Args:
        idx: 測試資料集中的圖片索引
        conf_thresh: 設定極低 (0.01) 以顯示被過濾掉的框
    """
    model.eval()
    
    # 1. 取得圖片與標註
    img_tensor, target = dataset[idx]
    image = img_tensor.to(device).unsqueeze(0) # [1, 3, H, W]
    
    # 2. 模型推論
    with torch.no_grad():
        # forward 回傳 (cls_logits, bbox_reg, centerness)
        cls_logits, bbox_reg, centerness = model(image)

    # --- [新增功能 1: 計算 Image-Level Confidence] ---
    # 這段邏輯與 calc_statistics 完全一致
    max_scores_per_level = []
    for i in range(len(cls_logits)):
        # 結合分數: [1, 1, H, W]
        level_score = (cls_logits[i].sigmoid() * centerness[i].sigmoid()).sqrt()
        # 展平並取該層最大值: [1, H*W] -> max(dim=1) -> [1]
        level_max = level_score.view(1, -1).max(dim=1)[0]
        max_scores_per_level.append(level_max)
    
    # 跨層級取最大值: [3, 1] -> max(dim=0) -> [1]
    # 這就是整張圖片最高的「瑕疵信心分數」
    image_confidence = torch.stack(max_scores_per_level, dim=0).max(dim=0)[0].item()
    # --------------------------------------------------
    
    # 3. 還原圖片 (Denormalize) 用於顯示
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    img_vis = img_np.copy() # 用於畫預測
    img_gt = img_np.copy()  # 用於畫 GT

    # 4. 畫 Ground Truth (綠色)
    gt_label = target['labels'].item()
    gt_text = "Normal" if gt_label == 0 else "Corrupted"
    
    if len(target['boxes']) > 0 and gt_label == 1:
        boxes = target['boxes'].cpu().numpy()
        for box in boxes:
            box = box.flatten()
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(img_gt, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(img_gt, "GT", (pt1[0], pt1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 在 GT 圖上標示該圖的真實類別
    cv2.putText(img_gt, f"GT: {gt_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 5. 解碼預測框 (使用傳入的閾值)
    strides = [8, 16, 32]
    outputs = (cls_logits, bbox_reg, centerness)
    
    # 引用 train.py 中的 post_process_fcos
    from train import post_process_fcos 
    # 注意：這裡 nms_thresh 設高一點 (例如 0.5)，避免太多重疊框
    results = post_process_fcos(outputs, strides, (288, 512), conf_thresh=conf_thresh, nms_thresh=0.5)
    pred = results[0]
    
    print(f"\n--- Debug Image {idx} ---")
    print(f"Image Confidence: {image_confidence:.4f}")
    print(f"GT Label: {gt_text} | Boxes: {len(target['boxes'])}")
    print(f"Pred Boxes (Thresh={conf_thresh}): {len(pred['boxes'])}")

    # 6. 畫預測框 (最高分紅色，其餘藍色)
    if len(pred['boxes']) > 0:
        # 找出最高分的索引
        max_score_idx = torch.argmax(pred['scores']).item()
        
        # 只畫前 10 個最強的，避免畫面太亂
        top_k = min(10, len(pred['boxes']))
        top_indices = torch.argsort(pred['scores'], descending=True)[:top_k]
        
        for i in top_indices:
            box = pred['boxes'][i].cpu().numpy()
            score = pred['scores'][i].item()
            
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            
            # --- [新增功能 2: 顏色區分] ---
            if i == max_score_idx:
                color = (255, 0, 0) # 最高分：紅色 (BGR)
                thickness = 3
            else:
                color = (0, 0, 255) # 其他：藍色 (BGR)
                thickness = 2
            # ---------------------------
            
            # 畫框
            cv2.rectangle(img_vis, pt1, pt2, color, thickness)
            
            # 顯示分數
            text = f"{score:.3f}"
            cv2.putText(img_vis, text, (pt1[0], pt1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if i == max_score_idx or i in top_indices[:3]:
                 print(f"  {'[MAX]' if i == max_score_idx else '     '} Box {i}: Score={score:.4f} Coord={box}")

    # 7. 繪製圖表
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_gt)
    plt.title(f"Ground Truth: {gt_text}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    # --- [新增功能 3: 標題顯示 Image Confidence] ---
    plt.imshow(img_vis)
    # --- [修改] 設定標題顏色邏輯 ---
    if image_confidence > 0.5:
        title_color = 'green' # 信心高 -> 綠色
    else:
        title_color = 'red'   # 信心低 -> 紅色
        
    plt.title(
        f"Preds (Thresh={conf_thresh})\nImage Conf: {image_confidence:.4f}", 
        color=title_color,      # 設定顏色
        fontweight='bold',      # 設定粗體
        fontsize=12             # 字體大小 (可選)
    )
    # ---------------------------
    # -------------------------------------------
    
    # 畫出 P4 層 (Stride 16) 的 Cls Heatmap
    cls_map = cls_logits[1][0].sigmoid().cpu().numpy().squeeze() # [H, W]
    
    plt.subplot(1, 3, 3)
    plt.imshow(cls_map, cmap='jet', vmin=0, vmax=1) # 固定 Heatmap 範圍 0~1
    plt.title("P4 Cls Heatmap (Sigmoid)")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f"./debugs9/debug_vis_{idx}.png", bbox_inches='tight', dpi=100) # 如果不想每次都存檔可以註解掉
    plt.close('all')
    # plt.show()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, delta=0.0, verbose=False):
        """
        Args:
            patience (int): 多少次 val_loss 沒改善就停止
            delta (float): 最小改善幅度 (改善需要 <= best_loss - delta)
            verbose (bool): 是否印出改善訊息
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        """每次 validation 結束後呼叫"""
        if val_loss < self.best_loss - self.delta:
            # Loss 有改善
            if self.verbose:
                print(f"Validation loss decreased: {self.best_loss:.6f} → {val_loss:.6f}. Saving model.")
            self.best_loss = val_loss
            self.counter = 0
        else:
            # 沒改善
            self.counter += 1
            if self.verbose:
                print(f"No improvement. EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
