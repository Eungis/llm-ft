from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any
from fox.base.meta import MetaSingleton
from fox.utils.constants import CONFIG_FILE_PATH, DB_TYPES

def load_yaml() -> dict:
    if not Path(CONFIG_FILE_PATH).exists():
        raise FileNotFoundError(f"Configuration file not found at the default path: {CONFIG_FILE_PATH}")

    with open(CONFIG_FILE_PATH) as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)
    return config

class FoxConfig(metaclass=MetaSingleton):
    
    def __init__(self, db: DB_TYPES):
        config = load_yaml()
        main_config = config.get("main")
        main_config.update(config.get(db))
        self.config = main_config
    
    @property
    def path(self) -> str:
        return CONFIG_FILE_PATH
    
    def read(self) -> dict:
        return self.config
    
    def get_key(self, k: str) -> Any:
        return self.config.get(k)
    
    def update_key(self, k: str, v: str):
        self.config[k] = v
        
        
        

    
    