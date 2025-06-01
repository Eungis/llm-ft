from typing import Literal

# constants variables
CONFIG_FILE_PATH = f"/usr/src/app/fox/conf/config.yaml"
DB_TYPES = Literal["elasticsearch", "chroma"]

# [TEMP] running variables
DB_TYPE = "elasticsearch"