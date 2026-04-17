"""
Sets up QoL environment variables
Usage:
    called by train.py
"""

import logging
import torch
import os
import warnings

def set_vars():
    # tracking fallback
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlruns.db"
    # disable warnings for cleaner CLI
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message=".*filesystem tracking backend.*")

    # possible speedups depending on user's HW/SW choices, refer to the PyTorch/CUDA docs
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = ".triton_cache"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TORCH_COMPILE_DEBUG"] = "0"
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(os.cpu_count())

    os.environ["NCCL_P2P_DISABLE"] = "0"
    os.environ["NCCL_IB_DISABLE"] = "1"