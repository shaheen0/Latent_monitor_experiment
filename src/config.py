from pathlib import Path
import torch
from transformers import AutoConfig  

print(f"pytorch version {torch.__version__}")
print(torch.cuda.get_arch_list())

if torch.cuda.is_available():
    print(f"Cuda/ROCm GPU: {torch.cuda.get_device_name(0)}")
else:
    print("only CPU")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
# model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
# model_id = "Qwen/Qwen3-0.6B"


# 1. Model ki config load karo (sirf layers count karne ke liye)
_model_config = AutoConfig.from_pretrained(model_id, local_files_only=False)
_num_layers = _model_config.num_hidden_layers

# 2. 75% layer nikaalo (Paper ke mutabiq)
LAYER_IDX = int(_num_layers * 0.75)

# 3. Model ka naam nikalo (e.g., "SmolLM2-360M-Instruct")
model_name = model_id.split("/")[-1]

print(f"elected Model: {model_id}")
print(f"Total Layers: {_num_layers}")
print(f"Target Layer (75%): {LAYER_IDX}")

# ============================================================
# Paths
# ============================================================
Base_dir = Path(__file__).resolve().parent.parent
Data_dir_read = Base_dir / "src" / "dataset"
OUTPUT_DIR = Base_dir / "data" / "splits"

# LAYER_DIR ab automatic model_name use karega (135M nahi!)
LAYER_DIR = (Base_dir / "outputs" / model_name / f"layer_{LAYER_IDX}")

ACT_DIR = (LAYER_DIR / "activations" / "probe_train" / "pooled")
ACTIVATIONS_DIR = LAYER_DIR / "activations"
PROBES_DIR = LAYER_DIR / "probes_filtered"

def pooled_dir(split: str) -> Path:
    return ACTIVATIONS_DIR / split / "pooled"

TRAIN_CONCEPTS: list[str] = [
    "all_caps",
    "biology",
    "chemistry",
    "comforting",
    "german",
    "history",
    "html",
    "jokey",
    "mathematics",
    "movie_review",
    "side_effects",
    "french",
    "spanish",
    "mandarin_chinese",
    "japanese",
    "arabic",
    "russian",
    "latin",
    "philosophy",
    "literature_focused",
    "linguistics",
    "music_theory",
    "religion_scripture",
    "mythology_folklore",
    "art_history",
    "cooking_recipes",
    "sports_commentary",
    "finance_economics", 
    "gardening", 
    "automotive_mechanical",
    "travel_writing", 
    "fashion", 
    "gaming", 
    "geology", 
    "computer_programming",
    "statistics", 
    "journalistic_headline",
]

TEST_CONCEPTS: list[str] = [
    "portuguese", "archaeology",
    "sarcastic", "sycophancy", "evasiveness_noncommittal",
    "imperatives_commands", "urgency_panic","deception", "physics","astronomy", 
]

