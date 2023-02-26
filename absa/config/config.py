import sys
import yaml
import os


class Config:
    def __init__(self):
        self.config_path = "absa/config/config.yaml"
        self.cfg = None

    def load_config(self):
        print(self.config_path, "ei")
        if not os.path.exists(self.config_path):
            print("congig_path: Path not exist {}".format(self.config_path))
            is_load_success = False
            return is_load_success
        try:
            with open(self.config_path, "r") as f:
                data = f.read()
                self.cfg = yaml.safe_load(data)
            is_load_success = True
        except Exception as e:
            print("load_config: Could not load file {}".format(e))
            is_load_success = False
        return is_load_success


if __name__ == "__main__":
    config = Config()
    config.load_config()
    print(config.cfg)
