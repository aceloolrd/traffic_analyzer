# main.py

import yaml
import argparse
from pipelines.traffic_pipeline import TrafficPipeline

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Traffic Analysis Pipeline")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the config file')
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = TrafficPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()
