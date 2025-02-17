from pathlib import Path
import yaml

def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent / "cfg.yaml"

    if not Path(config_path).exists():
        raise FileNotFoundError(f"Файл конфигурации {config_path} не найден.")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Ошибка при загрузке YAML-файла {config_path}: {e}")

    return config

train_config = load_config()
