# -*- coding: utf-8 -*-
import os
import sys
import argparse
import torch
from tqdm import tqdm

# ===================== 路径修复 =====================
# 把 uie 子目录加入 Python 路径，确保能找到 datasets 等模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'uie'))

# ===================== 导入模块 =====================
# 注意：dataset.py 里的类名是 ImageDataset，不是 FlowerDataset
from datasets.dataset import ImageDataset
from datasets.transforms import get_val_transforms
from models.vit import create_vit_model
from utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description="批量验证脚本（适配项目结构）")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="模型权重路径，默认加载 work_dirs/best.pt")
    args = parser.parse_args()

    # ===================== 1. 加载配置 =====================
    # 你的 load_config 不需要传参，直接调用
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = cfg["data"]["num_classes"]
    class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]  # 5类花朵名称

    # ===================== 2. 构建模型 & 加载权重 =====================
    model = create_vit_model(num_classes=num_classes).to(device)
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(cfg["train"]["save_path"], "best.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"权重文件未找到: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"✅ 成功加载权重: {checkpoint_path}")

    # ===================== 3. 构建验证集 DataLoader =====================
    val_transform = get_val_transforms(cfg["data"]["image_size"])
    val_dataset = ImageDataset(
        data_root=cfg["data"]["val_root"],
        transforms=val_transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"]
    )

    # ===================== 4. 批量推理 & 计算准确率 =====================
    total_correct = 0
    total_num = 0

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="🔍 正在验证"):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            total_correct += (preds == labels).sum().item()
            total_num += imgs.size(0)

    val_acc = total_correct / total_num
    print(f"\n=== 验证结果 ===")
    print(f"📊 总样本数: {total_num}")
    print(f"🎯 整体准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")

if __name__ == "__main__":
    main()