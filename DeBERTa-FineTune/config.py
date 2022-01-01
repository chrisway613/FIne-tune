"""
    Configurations of DeBERTa Fine-tuning task.
"""

import os


# Random seed
SEED=42

# Pre-trained model
RESUME = False
MODEL_CHECKPOINT = 'microsoft/deberta-large'

# Squad dataset version: 1/2
SQUAD_VER = 1
# Dataset name
# (for SQUADv1.1, the data files will be cached in:
#  ~/.cache/huggingface/datasets/squad/plain_text/1.0.0/)
DATASET_NAME = 'squad' if SQUAD_VER == 1 else 'squad_v2'
# Dataset local path (if already downloaded)
DATASET_PATH = '/home/weicai/.cache/huggingface/datasets/squad/plain_text/1.0.0/d6ec3ceb99ca480ce37cdd35555d6cb2511d223b9150cce08a837ef62ffea453'

# Max sequence length
MAX_LENGTH = 384
# Best answer candidates
N_BEST_ANSWERS = 20
# Max length of answer
MAX_ANSWER_LENGTH = 30
# Overlap length between spans
DOC_STRIDE = 128
# We need to account for the special case where the model expects padding on the left
# (in which case we switch the order of the question and the context)
PAD_ON_RIGHT = True

LR = 5e-6
EPOCHS = 10
BATCH_SIZE = 16
WARMUP_STEPS = 50

# Project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Directory which to save output results
OUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Tensor format: 'pt', 'tf', etc
TENSOR_FORMAT = 'pt'
# Data format: None, 'numpy', 'torch', 'tensorflow', 'pandas', 'arrow'
DATA_FORMAT = 'torch'