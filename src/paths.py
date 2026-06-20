from pathlib import Path


# Repository paths
REPO_ROOT = Path(__file__).resolve().parents[1]


# Cache and tool configuration paths - needed for ranx to work
CACHE_DIR = REPO_ROOT / ".cache"
IR_DATASETS_HOME = CACHE_DIR / "ir_datasets"
MATPLOTLIB_CONFIG_DIR = CACHE_DIR / "matplotlib"


# Shared data paths
DATA_DIR = REPO_ROOT / "data"
INDICES_DIR = DATA_DIR / "indices"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
TEST_DIR = DATA_DIR / "test"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
TEST_QUERIES_PATH = TEST_DIR / "queries.jsonl"


# Run artifact paths
RUNS_DIR = REPO_ROOT / "runs"
RETRIEVAL_RUNS_DIR = RUNS_DIR / "retrieval"
