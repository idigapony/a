import sys
import os
# 把最外层 uie/ 目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os
os.environ['ALBUMENTATIONS_SKIP_VERSION_CHECK'] = '1'

import torch
from torch.utils.data import DataLoader
from uie.datasets.dataset import ImageDataset
from uie.models.vit import create_vit_model
from uie.losses.loss import get_criterion
from uie.optim.optimizer import get_optimizer
from uie.optim.scheduler import get_scheduler
from uie.utils.config import load_config
from uie.utils.metrics import calculate_accuracy
from uie.utils.logger import Logger


def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["train"]["save_path"], exist_ok=True)

    train_dataset = ImageDataset(cfg["data"]["root"], mode="train", img_size=cfg["data"]["image_size"])
    val_dataset = ImageDataset(cfg["data"]["root"], mode="val", img_size=cfg["data"]["image_size"])

    train_loader = DataLoader(train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=cfg["train"]["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            num_workers=cfg["train"]["num_workers"])

    model = create_vit_model(num_classes=cfg["model"]["num_classes"]).to(device)
    criterion = get_criterion()
    optimizer = get_optimizer(model, lr=cfg["train"]["lr"])
    scheduler = get_scheduler(optimizer)

    logger = Logger(cfg["train"]["save_path"])
    best_acc = 0.0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_acc += calculate_accuracy(outputs, labels) * imgs.size(0)

        train_loss /= len(train_dataset)
        train_acc /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item() * imgs.size(0)
                val_acc += calculate_accuracy(outputs, labels) * imgs.size(0)

        val_loss /= len(val_dataset)
        val_acc /= len(val_dataset)
        scheduler.step()

        logger.log(epoch+1, train_loss, train_acc, val_loss, val_acc)
        print(f"Epoch {epoch+1:02d} | Train L:{train_loss:.3f} A:{train_acc:.3f} | Val L:{val_loss:.3f} A:{val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "best_acc": best_acc
            }, os.path.join(cfg["train"]["save_path"], "best.pth"))

    logger.save()
    print(f"Best val acc: {best_acc:.3f}")


if __name__ == "__main__":
    main()