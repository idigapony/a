# DeiT-S 花朵图像分类项目
基于 PyTorch 实现的 DeiT-S 模型花朵分类解决方案，覆盖训练、批量验证、单图推理全流程，工程结构规范，性能指标突出。

# 项目简介
本项目采用 DeiT-S (Data-efficient Image Transformer Small) 轻量化视觉 Transformer 模型，在 5 类花朵数据集（雏菊、蒲公英、玫瑰、向日葵、郁金香）上完成微调训练。通过精细化的数据增强策略与动态学习率调度，最终在 865 个验证样本上实现 95.49% 的分类准确率，兼顾模型泛化能力与工程落地性。

# 项目结构
uie/
├── test.jpg                  # 测试样例图片
├── test.py                   # 单图推理脚本
├── train.py                  # 训练主脚本
├── val.py                    # 批量验证脚本
│
├── data/                     # 数据集目录
│   └── flowers/              # 5类花朵数据集
│       ├── train/            # 训练集（按类别分文件夹）
│       └── val/              # 验证集（按类别分文件夹）
│
├── uie/                      # 核心代码包
│   ├── datasets/             # 数据处理模块
│   ├── models/               # 模型定义模块
│   ├── configs/              # 配置文件模块
│   ├── losses/               # 损失函数模块
│   ├── optim/                # 优化器与调度器
│   └── utils/                # 工具函数
│
└── work_dirs/                # 训练输出目录
    ├── best.pt               # 验证集最优权重
    ├── epoch_*.pt            # 各轮次检查点权重
    └── log.txt               # 训练日志

# 核心亮点
1. 高性能指标：在 865 个验证样本上实现 95.49% 分类准确率，模型泛化能力优异。
2. 轻量化模型选型：采用 DeiT-S 预训练模型，兼顾算力效率与分类精度，适配小样本场景微调。
3. 完善的数据增强：集成 Mosaic 拼接、随机裁剪 / 翻转 / 旋转、颜色抖动等策略，有效缓解小数据集过拟合。
4. 成熟的训练策略：使用 Warmup + 余弦退火学习率调度，配合 AdamW 优化器，实现模型稳定收敛。
5. 全流程工程化：训练、验证、推理脚本解耦且联动，支持权重加载、日志记录、批量评估，可直接复现实验。

# 环境依赖
1. torch>=2.0.0
2. torchvision>=0.15.0
3. numpy>=1.24.0
4. tqdm>=4.66.0
5. pyyaml>=6.0.1
6. albumentations>=1.4.0
7. opencv-python>=4.9.0
8. Pillow>=10.0.0
一键安装命令：pip install -r requirements.txt

# 快速开始
1. 克隆项目
git clone <https://github.com/idigapony/a>
cd uie
2. 安装依赖
pip install -r requirements.txt
3. 模型训练
python train.py
训练过程中自动完成：
每轮验证并保存最优权重 work_dirs/best.pt
日志持久化至 work_dirs/log.txt
按 5 轮间隔保存检查点权重
4. 批量验证
验证最优权重
python val.py --checkpoint ./work_dirs/best.pt
验证指定轮次权重
python val.py --checkpoint ./work_dirs/epoch_30.pt
5. 单图推理
python test.py --img_path ./test.jpg --checkpoint ./work_dirs/best.pt

# 实验结果
验证集样本总数：865
最终分类准确率：95.49%
训练总轮次：40 轮
最优权重对应轮次：第 39 轮

# 许可证
MIT License
