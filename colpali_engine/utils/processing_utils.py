
import collections
import torch
from PIL import Image
from tqdm import tqdm
from abc import ABC, abstractmethod
from multiprocessing import Pool
from typing import List, Dict, Any, Tuple, Optional, Union
from transformers import BatchEncoding, BatchFeature, ProcessorMixin

from colpali_engine.utils.torch_utils import get_torch_device


class BaseVisualRetrieverProcessor(ABC, ProcessorMixin):
    """
    Base class for visual retriever processors.
    """

    @abstractmethod
    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        pass

    @abstractmethod
    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> Union[BatchFeature, BatchEncoding]:
        pass

    @abstractmethod
    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def score_single_vector(
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the dot product score for the given single-vector query and passage embeddings.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        qs_stacked = torch.stack(qs).to(device)
        ps_stacked = torch.stack(ps).to(device)

        scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @staticmethod
    def score_multi_vector(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
        query embeddings (`qs`) and passage embeddings (`ps`). For ColPali, a passage is the
        image of a document page.

        Because the embedding tensors are multi-vector and can thus have different shapes, they
        should be fed as:
        (1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
        (2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
            obtained by padding the list of tensors.

        Args:
            qs (`Union[torch.Tensor, List[torch.Tensor]`): Query embeddings.
            ps (`Union[torch.Tensor, List[torch.Tensor]`): Passage embeddings.
            batch_size (`int`, *optional*, defaults to 128): Batch size for computing scores.
            device (`Union[str, torch.device]`, *optional*): Device to use for computation. If not
                provided, uses `get_torch_device("auto")`.

        Returns:
            `torch.Tensor`: A tensor of shape `(n_queries, n_passages)` containing the scores. The score
            tensor is saved on the "cpu" device.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        scores_list: List[torch.Tensor] = []

        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(qs[i : i + batch_size], batch_first=True, padding_value=0).to(
                device
            )
            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0
                ).to(device)
                scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)

        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @staticmethod
    def _find_connected_regions(points, neighbor_range=[-1, 0, 1]):
        points_set = set((x, y) for x, y in points)
        visited = set()
        regions = []
        
        for point in points_set:
            if point not in visited:
                queue = collections.deque()
                queue.append(point)
                visited.add(point)
                region = []
                
                while queue:
                    x, y = queue.popleft()
                    region.append((x, y))
                    
                    # 检查八个方向的邻居
                    for dx in neighbor_range:
                        for dy in neighbor_range:
                            if dx == 0 and dy == 0:
                                continue  # 跳过当前点本身
                            nx = x + dx
                            ny = y + dy
                            neighbor = (nx, ny)
                            
                            if neighbor in points_set and neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)
                
                regions.append(region)
        
        regions = [torch.tensor(region) for region in regions]
        return regions

    @staticmethod
    def _calculate_bounding_boxes(regions, image_size, grid_size):
        height, width = image_size
        grid_h, grid_w = grid_size
        bounding_boxes = []
        
        for region in regions:
            
            # 将区域中的网格坐标转为numpy数组
            # grid_coords = np.array(region)
            grid_x = region[:, 1]
            grid_y = region[:, 0]
            
            # 计算极值索引
            min_x, max_x = torch.min(grid_x).item(), torch.max(grid_x).item()
            min_y, max_y = torch.min(grid_y).item(), torch.max(grid_y).item()
            
            # 计算实际坐标（处理边界）
            x1 = min_x * grid_w
            y1 = min_y * grid_h
            x2 = min((max_x + 1) * grid_w, width)
            y2 = min((max_y + 1) * grid_h, height)
            
            bounding_boxes.append((x1, y1, x2, y2))
        
        return bounding_boxes

    @staticmethod
    def score_multi_vector_per_patch(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 16,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Compute the late-interaction/MaxSim score (ColBERT-like) for the given multi-vector
        query embeddings (`qs`) and passage embeddings (`ps`). For ColPali, a passage is the
        image of a document page.

        Because the embedding tensors are multi-vector and can thus have different shapes, they
        should be fed as:
        (1) a list of tensors, where the i-th tensor is of shape (sequence_length_i, embedding_dim)
        (2) a single tensor of shape (n_passages, max_sequence_length, embedding_dim) -> usually
            obtained by padding the list of tensors.

        Args:
            qs (`Union[torch.Tensor, List[torch.Tensor]`): Query embeddings.
            ps (`Union[torch.Tensor, List[torch.Tensor]`): Passage embeddings.
            batch_size (`int`, *optional*, defaults to 128): Batch size for computing scores.
            device (`Union[str, torch.device]`, *optional*): Device to use for computation. If not
                provided, uses `get_torch_device("auto")`.

        Returns:
            `torch.Tensor`: A tensor of shape `(n_queries, n_passages)` containing the scores. The score
            tensor is saved on the "cpu" device.
        """
        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")
        
        scores_list: List[torch.Tensor] = []

        # Pad passage embeddings for batching
        ps_padded = torch.nn.utils.rnn.pad_sequence(ps, batch_first=True, padding_value=0)
        p_mask = ps_padded.abs().sum(dim=-1) > 0 # Mask for actual patches (not padding)
        scores_list = []
        for i in range(0, len(qs), batch_size):
            # Pad query embeddings for batching
            qs_batch = torch.nn.utils.rnn.pad_sequence(
                qs[i : i + batch_size], batch_first=True, padding_value=0
            ).to(device)
            # Create mask for actual query tokens (not padding)
            q_mask_batch = qs_batch.abs().sum(dim=-1) > 0 # (batch_q, seq_len_q)
            scores_batch = []
            for j in range(0, len(ps), batch_size):
                ps_batch = ps_padded[j : j + batch_size].to(device)
                scores_batch.append(
                    torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).sum(dim=2) / 
                    q_mask_batch.unsqueeze(1).int().sum(dim=-1, keepdim=True)
                )
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)

        scores = torch.cat(scores_list, dim=0)
        scores = scores.to(torch.float32)

        return scores, p_mask

    @staticmethod
    def single_get_box(args):
        query_id, image_id, score, score_method, threshold, neighbor_range, image_size, image_grid, grid_size = args
        # Find indices of patches above the threshold
        selected_indices = torch.where(score > threshold)[0]

        if selected_indices.numel() == 0:
            return [] # No patches above threshold for this image
        
        selected_h = selected_indices // image_grid[2]
        selected_w = selected_indices % image_grid[2]
        selected_2d_indices = torch.stack([selected_h, selected_w], dim=-1).tolist()

        # Find connected components among selected patches
        connected_2d_indices = BaseVisualRetrieverProcessor._find_connected_regions(selected_2d_indices, neighbor_range)

        bounding_box = BaseVisualRetrieverProcessor._calculate_bounding_boxes(connected_2d_indices, image_size, grid_size)
        score_2d = score.reshape(image_grid[1], image_grid[2])
        if score_method == "max":
            bounding_box_score = [score_2d[index[:, 0], index[:, 1]].max().item() for index in connected_2d_indices]
        elif score_method == "mean":
            bounding_box_score = [score_2d[index[:, 0], index[:, 1]].mean().item() for index in connected_2d_indices]


        results = [{"query_id": query_id, "image_id": image_id, "bounding_box": box, "score": score} for score, box in zip(bounding_box_score, bounding_box)]
        return results

    @abstractmethod
    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        *args,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an
        image of size (height, width) with the given patch size.
        """
        pass
