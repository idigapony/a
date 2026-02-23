import torch.nn as nn
import torchvision.models as models


def create_vit_model(num_classes=5):
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model