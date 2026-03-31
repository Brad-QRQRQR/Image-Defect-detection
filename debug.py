import torch
from dataset import CorruptionLMDBDataset
from train import prepare
from utils import visualize_debug

if __name__ == "__main__":
    test_dataset = CorruptionLMDBDataset(
        lmdb_path=r"D:\comVis\content\test_lmdb",
        target_size=(512, 288),
    )
    model, history, device = prepare()
    model.load_state_dict(torch.load(r".\fcol_320by320_final9\best_model_map.pth"))
    for i in range(len(test_dataset) - 1, -1, -1):
        visualize_debug(model, test_dataset, device, idx=i)
    # 7882