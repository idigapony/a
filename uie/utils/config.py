import yaml
import os


def load_config():
    # 从当前脚本位置自动找到 config.yaml
    current_file = os.path.abspath(__file__)
    utils_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(utils_dir)  # 即 uie/
    config_path = os.path.join(project_root, "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg