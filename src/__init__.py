import os

from src.paths import IR_DATASETS_HOME, MATPLOTLIB_CONFIG_DIR

IR_DATASETS_HOME.mkdir(parents=True, exist_ok=True)
MATPLOTLIB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("IR_DATASETS_HOME", str(IR_DATASETS_HOME))
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_DIR))
