import json
import os
import time


class Logger:
    def __init__(self, save_path):
        self.history = []
        self.save_path = save_path
        self.start_time = time.time()

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc):
        elapsed = time.time() - self.start_time
        self.history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "time_sec": round(elapsed, 2)
        })

    def save(self):
        os.makedirs(self.save_path, exist_ok=True)
        path = os.path.join(self.save_path, "train.log")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)