import os

import lmdb
import torch
import io
import numpy as np

from collections import defaultdict
from torch.utils.data import Dataset

from tqdm import tqdm

import pickle

from torchvision.transforms import v2
from torchvision import tv_tensors

from PIL import Image

def write_to_lmdb(data_list, lmdb_path, map_size):
    """
    負責將資料列表寫入 LMDB 的通用函數
    """
    if os.path.exists(lmdb_path):
        print(f"警告: {lmdb_path} 已存在，建議先刪除舊檔以免混亂。")
        return

    print(f"正在寫入 {lmdb_path}，共 {len(data_list)} 筆資料...")

    env = lmdb.open(lmdb_path, map_size=map_size)

    with env.begin(write=True) as txn:
        for idx, info in enumerate(tqdm(data_list)):

            # 讀取圖片原始 Bytes
            try:
                with open(info['full_img_path'], 'rb') as f:
                    image_bytes = f.read()
            except FileNotFoundError:
                print(f"遺失圖片: {info['full_img_path']}")
                continue

            assert len(info['bbox']) == 4, f"bbox 格式錯誤: {info['bbox']}"
            # 準備儲存物件
            sample = {
                'image': image_bytes,
                'label': info['label'],
                'bbox': info['bbox'],
                'type': info['corruption_type'],
                'video_id': info['video_id'] # 順便存影片 ID，方便之後除錯
            }

            key = str(idx).encode('ascii')
            txn.put(key, pickle.dumps(sample))

        # 紀錄總長度
        txn.put('length'.encode('ascii'), str(len(data_list)).encode('ascii'))

    print(f"寫入完成！位置: {lmdb_path}")

def create_split_lmdb(root_dir, output_dir, split_ratio=0.8):
    """
    root_dir: 原始資料根目錄
    output_dir: 輸出 LMDB 的父目錄
    split_ratio: 訓練集比例 (0.8 代表 8:2 分割)
    """

    sub_folders = ['blocks', 'ghosting', 'mosaic', 'tear']

    # 用來暫存分好組的資料: grouped_data[瑕疵類型][影片ID] = [Frame1, Frame2...]
    grouped_data = defaultdict(lambda: defaultdict(list))

    print("--- 步驟 1: 掃描並依照影片分組 ---")

    total_frames = 0
    for sub_folder in sub_folders:
        label_file = os.path.join(root_dir, sub_folder, "labels_bbox.txt")

        if not os.path.exists(label_file):
            print(f"找不到 {label_file}，跳過。")
            continue

        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(' ')
                if len(parts) < 2:
                  print(f"警告: {sub_folder} 底下 {line} 存在格式錯誤")
                  continue

                # parts[0] 範例: 720p_240fps_1/frame_000003.jpg
                img_rel_path = parts[0]
                label = int(parts[1])

                # 抓出影片 ID (例如: 720p_240fps_1)
                video_id = img_rel_path.split('/')[0]
                bbox = [float(p) for p in parts[2:]] if len(parts) >= 6 else [0.0, 0.0, 0.0, 0.0]

                full_img_path = os.path.join(root_dir, sub_folder, "images", img_rel_path)

                item = {
                    'full_img_path': full_img_path,
                    'label': label,
                    'bbox': bbox,
                    'corruption_type': sub_folder,
                    'video_id': video_id
                }

                # 存入字典結構
                grouped_data[sub_folder][video_id].append(item)
                total_frames += 1

    print(f"總共掃描到 {total_frames} 幀圖片。")

    print("--- 步驟 2: 執行影片級別的 Train/Test 分割 ---")

    final_train_list = []
    final_test_list = []

    # 設定種子確保每次跑結果一樣
    np.random.seed(42)

    # 針對每一種瑕疵類型，分開做切割，確保 Test Set 裡每種瑕疵都有
    for c_type in sub_folders:
        videos_dict = grouped_data[c_type]
        video_ids = list(videos_dict.keys())

        # 打亂影片順序
        np.random.shuffle(video_ids)

        # 計算切分點
        split_idx = int(len(video_ids) * split_ratio)

        train_vids = video_ids[:split_idx]
        test_vids = video_ids[split_idx:]

        print(f"類別 [{c_type}]: 總影片 {len(video_ids)} 支 -> Train: {len(train_vids)}, Test: {len(test_vids)}")

        # 將對應影片的所有 Frames 加入最終列表
        for vid in train_vids:
            final_train_list.extend(videos_dict[vid])

        for vid in test_vids:
            final_test_list.extend(videos_dict[vid])

    print(f"分割總結 -> Train 總張數: {len(final_train_list)}, Test 總張數: {len(final_test_list)}")

    # --- 步驟 3: 寫入 LMDB ---
    # 預估空間: 1TB (不會真的佔用，只是 Map Size)
    map_size = (1024 ** 3) * 10

    train_lmdb_path = os.path.join(output_dir, "train_lmdb")
    test_lmdb_path = os.path.join(output_dir, "test_lmdb")

    write_to_lmdb(final_train_list, train_lmdb_path, map_size)
    write_to_lmdb(final_test_list, test_lmdb_path, map_size)

class CorruptionLMDBDataset(Dataset):
    def __init__(self, lmdb_path, transform=None, target_size=(128, 128)):
        """
        Args:
            lmdb_path (str): LMDB 資料夾路徑
            target_size (tuple): (W, H) 模型輸入大小，預設 (128, 128)
            transform (callable, optional): 額外的影像增強 (如 ColorJitter)
        """
        self.lmdb_path = lmdb_path
        self.target_size = target_size
        self.env = None # 延遲初始化，防止多行程錯誤

        self.base_transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.augment_transform = transform # 外部傳入的增強 (如 ColorJitter)

    def _init_db(self):
        """在 Worker 內部初始化 LMDB 環境"""
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            length_bytes = txn.get('length'.encode('ascii'))
            self.length = int(length_bytes.decode('ascii'))

    def __getstate__(self):
        state = self.__dict__.copy()
        state['env'] = None  # 關鍵：序列化時強制把 env 設為 None
        return state

    def __len__(self):
        if self.env is None:
            self._init_db()
        return self.length

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()

        # 1. 讀取資料
        with self.env.begin(write=False) as txn:
            key = str(idx).encode('ascii')
            byteflow = txn.get(key)

        sample = pickle.loads(byteflow)

        # image_bytes -> PIL Image
        image = Image.open(io.BytesIO(sample['image'])).convert('RGB')

        # 2. 取得原始資訊
        w_old, h_old = image.size # 原始圖片尺寸
        raw_bbox = sample['bbox'] # 原始座標 [x, y, w, h]
        label = sample['label']

        # 3. 圖片處理 (Resize)
        # 注意：這裡我們先做 Resize，再轉 Tensor
        image = image.resize(self.target_size)

        img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)

        # ★ 修改開始 ★
        # 這裡不要除以 w_old，而是要將座標「縮放」到 224x224 的尺度
        # 因為你的圖片已經 resize 到 target_size (224) 了，BBox 也要跟著縮放到 224
        if label == 1:
            scale_x = self.target_size[0] / w_old
            scale_y = self.target_size[1] / h_old
            
            abs_bbox = [
                raw_bbox[0] * scale_x, # x (0~224)
                raw_bbox[1] * scale_y, # y (0~224)
                raw_bbox[2] * scale_x, # w (0~224)
                raw_bbox[3] * scale_y  # h (0~224)
            ]
            
            # 傳入絕對座標，canvas_size 設定正確，這樣 Flip 才會算出正確的數值
            boxes = tv_tensors.BoundingBoxes(
                [abs_bbox], 
                format=tv_tensors.BoundingBoxFormat.XYWH, 
                canvas_size=(self.target_size[1], self.target_size[0])
            )
        else:
            abs_bbox = [0.0, 0.0, 0.0, 0.0]
            boxes = tv_tensors.BoundingBoxes(
                [abs_bbox], 
                format=tv_tensors.BoundingBoxFormat.XYWH, 
                canvas_size=(self.target_size[1], self.target_size[0])
            )

        if not self.augment_transform:
            # final_bbox = boxes[0] / self.target_size[0] # 假設長寬相等，簡化寫法
            final_box = boxes[0] # [x, y, w, h]
            if label == 1:
                # 轉為 x1, y1, x2, y2
                converted_box = torch.tensor([
                    final_box[0],               # x1
                    final_box[1],               # y1
                    final_box[0] + final_box[2], # x2 = x + w
                    final_box[1] + final_box[3]  # y2 = y + h
                ], dtype=torch.float32)
            else:
                converted_box = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            return self.base_transform(img_tensor), {'labels': torch.tensor([label], dtype=torch.long), 'boxes': converted_box.unsqueeze(0)}

        # 5. 應用 Transform (這時候傳進去的是 0~224 的座標)
        img_aug, boxes_aug = self.augment_transform(img_tensor, boxes)

        if label == 1 and len(boxes_aug) > 0:
            # ★ Transform 完之後，再 Normalize 到 0~1 ★
            # 因為 RandomHorizontalFlip 會在 0~224 的範圍內翻轉，翻完還是 0~224
            final_box = boxes_aug[0]
            # final_bbox = final_bbox / self.target_size[0]
            converted_box = torch.tensor([
                final_box[0],
                final_box[1],
                final_box[0] + final_box[2],
                final_box[1] + final_box[3]
            ], dtype=torch.float32)
        else:
            converted_box = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

        return img_aug, {'labels': torch.tensor([label], dtype=torch.long), 'boxes': converted_box.unsqueeze(0)}