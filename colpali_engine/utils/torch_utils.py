import gc
import logging
from typing import List, TypeVar
import os
import pathlib
import shutil
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
T = TypeVar("T")


def get_torch_device(device: str = "auto") -> str:
    """
    Returns the device (string) to be used by PyTorch.

    `device` arg defaults to "auto" which will use:
    - "cuda:0" if available
    - else "mps" if available
    - else "cpu".
    """

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():  # for Apple Silicon
            device = "mps"
        else:
            device = "cpu"
        logger.info(f"Using device: {device}")

    return device


def tear_down_torch():
    """
    Teardown for PyTorch.
    Clears GPU cache for both CUDA and MPS.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


class ListDataset(Dataset[T]):
    def __init__(self, elements: List[T]):
        self.elements = elements

    def __len__(self) -> int:
        return len(self.elements)

    def __getitem__(self, idx: int) -> T:
        return self.elements[idx]


def unbind_padded_multivector_embeddings(
    embeddings: torch.Tensor,
    padding_value: float = 0.0,
    padding_side: str = "left",
) -> List[torch.Tensor]:
    """
    Removes padding elements from a batch of multivector embeddings.

    Args:
        embeddings (torch.Tensor): A tensor of shape (batch_size, seq_length, dim) with padding.
        padding_value (float): The value used for padding. Each padded token is assumed
            to be a vector where every element equals this value.
        padding_side (str): Either "left" or "right". This indicates whether the padded
            elements appear at the beginning (left) or end (right) of the sequence.

    Returns:
        List[torch.Tensor]: A list of tensors, one per sequence in the batch, where
            each tensor has shape (new_seq_length, dim) and contains only the non-padding elements.
    """
    results: List[torch.Tensor] = []

    for seq in embeddings:
        is_padding = torch.all(seq.eq(padding_value), dim=-1)

        if padding_side == "left":
            non_padding_indices = (~is_padding).nonzero(as_tuple=False)
            if non_padding_indices.numel() == 0:
                valid_seq = seq[:0]
            else:
                first_valid_idx = non_padding_indices[0].item()
                valid_seq = seq[first_valid_idx:]
        elif padding_side == "right":
            non_padding_indices = (~is_padding).nonzero(as_tuple=False)
            if non_padding_indices.numel() == 0:
                valid_seq = seq[:0]
            else:
                last_valid_idx = non_padding_indices[-1].item()
                valid_seq = seq[: last_valid_idx + 1]
        else:
            raise ValueError("padding_side must be either 'left' or 'right'.")
        results.append(valid_seq)

    return results


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def is_ckpt_valid(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return False
    # if not os.path.exists(os.path.join(ckpt_dir, "config.json")):
    #     return False
    # if not os.path.exists(os.path.join(ckpt_dir, "tokenizer.json")):
    #     return False
    if len(list(pathlib.Path(ckpt_dir).glob("*.safetensors"))) == 0:
        return False
    if not os.path.exists(os.path.join(ckpt_dir, "scheduler.pt")):
        return False
    if not os.path.exists(os.path.join(ckpt_dir, "trainer_state.json")):
        return False
    if not os.path.exists(os.path.join(ckpt_dir, "training_args.bin")):
        return False
    if len(list(pathlib.Path(ckpt_dir).glob("rng_state*.pth"))) == 0:
        return False

    return True


def check_ckpt_exists(output_dir):
    if is_main_process(): # 分布式训练中一般只有主进程(rank=0)负责检查和清理checkpoint，避免多个进程同时操作导致文件冲突
        while list(pathlib.Path(output_dir).glob("checkpoint-*")): # 反复检查最新的 checkpoint 是否有效，不行就删除，直到找到一个有效的为止，或者没有 checkpoint
            ckpt_paths = list(pathlib.Path(output_dir).glob("checkpoint-*")) # 把所有匹配 checkpoint-* 的路径收集到列表里
            ckpt_iters = [int(path.name.split('checkpoint-')[-1]) for path in ckpt_paths] # 遍历每个ckpt名字，获取其最大的迭代次数
            max_ckpt_iter = max(ckpt_iters)
            max_ckpt_path = ckpt_paths[ckpt_iters.index(max_ckpt_iter)]
            if is_ckpt_valid(max_ckpt_path): # 检查最新的 checkpoint 是否完整/可用，如果可用就跳出循环
                break
            shutil.rmtree(max_ckpt_path) # 否则就删除最新的ckpt（无效的ckpt）
            print("removed invalid checkpoint: ", max_ckpt_path)
    
    synchronize() # 将ckpt信息同步所有进程
    if list(pathlib.Path(output_dir).glob("checkpoint-*")):
        return True
    else:
        return False

