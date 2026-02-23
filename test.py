import sys
import os
# 把最外层 uie/ 目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import cv2
from uie.datasets.dataset import ImageDataset
from uie.models.vit import create_vit_model
from uie.utils.config import load_config
from uie.utils.metrics import calculate_accuracy
from uie.datasets.transforms import get_val_transforms


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = create_vit_model(num_classes=cfg["model"]["num_classes"]).to(device)
    checkpoint_path = os.path.join(cfg["train"]["save_path"], "best.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 测试单张图片
    img_path = "./test.jpg"  # 绝对路径：uie/test.jpg
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = get_val_transforms(cfg["data"]["image_size"])
    img = transform(image=img)["image"]
    img = img.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, dim=1)

    classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
    print(f"Predicted class: {classes[pred.item()]}")


if __name__ == "__main__":
    main()