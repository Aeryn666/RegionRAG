from typing import Any, Dict, List, Union, cast
import torch
from PIL.Image import Image
from PIL import Image as PILImage

from colpali_engine.models.paligemma import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


def prefix_keys(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Prefix all keys in a dictionary with the given prefix.
    """
    return {f"{prefix}{k}": v for k, v in data.items()}


class VisualRetrieverCollator:
    """
    Collator for training vision retrieval models.
    """

    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_token_id = None

        # If processor is one of the supported types, extract the <image> token id.
        if isinstance(self.processor, ColPaliProcessor):
            image_token = "<image>"
            try:
                idx = self.processor.tokenizer.additional_special_tokens.index(image_token)
                self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[idx]
            except ValueError:
                self.image_token_id = None

        # Force padding to be on the right for ColPaliProcessor.
        if isinstance(self.processor, ColPaliProcessor) and self.processor.tokenizer.padding_side != "right":
            print("Setting padding side to right")
            self.processor.tokenizer.padding_side = "right"

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts_query: List[Union[str, None]] = []
        images: List[Image] = []
        neg_images: List[Image] = []
        bboxes: List[Union[List[int], None]] = []

        # Parse the examples.
        for example in examples:
            query = example.get("query")
            texts_query.append(query)

            image = example.get("image", None)
            image_path = example.get("image_path", None)
            if isinstance(image, str):  # 如果是 image_path 字符串
                image = PILImage.open(image).convert("RGB")
            if image is None:
                assert image_path is not None
                image = PILImage.open(image_path).convert("RGB")
                # raise ValueError("Image is None - This collator does not support None images yet.")
            images.append(cast(Image, image)) # Cast(Type, Value)意思是把Value转换成Type类型来看，它不会真的改变Value的类型，只是返回原值

            neg_image = example.get("neg_image")
            if neg_image is not None:
                neg_images.append(cast(Image, neg_image))

            bbox = example.get("bbox", None)
            if bbox is not None:    # x1, y1, x2, y2
                x1, y1, x2, y2 = bbox
                bbox = [
                    x1 / image.width,
                    y1 / image.height,
                    x2 / image.width,
                    y2 / image.height,
                ]
            bboxes.append(bbox)
            
        # Process images.
        batch_doc = self.processor.process_images(images=images)
        batch_neg_doc = self.processor.process_images(images=neg_images) if neg_images else None

        image_masks = batch_doc['input_ids'] == self.processor.image_token_id
        bbox_masks = []

        if "image_grid_thw" in batch_doc:
            # 如果processor返回了image_grid_thw，说明此时用的是Qwen的processor，可以直接用这个image_grid_thw
            grids = batch_doc['image_grid_thw']
        else:
            # 否则用pixel_values来计算grid
            grids = []
            for pixel_value in batch_doc['pixel_values']:
                _, H, W = pixel_value.shape
                h = torch.tensor(H) // 14 # Colpali的base_model_name_or_path是colpaligemma-3b-pt-448-base，其config文件里面写的patch_size是14，所以这里要除以14
                w = torch.tensor(W) // 14 # 输入的图片在统一处理后是448*448，所以最后得到的token_grid的大小是（448/14）*（448/14）=32*32=1024），这与我们前面在PaliGemmaProcessor类看到的self.image_seq_length=1024一致
                t = 1
                grids.append((t, h, w))

        for bbox, grid, image_mask in zip(bboxes, grids, image_masks):
            t, h, w = grid
            if not isinstance(self.processor, ColPaliProcessor):
                h = h // 2
                w = w // 2 # 因为qwen2.5vl的ViT里面有一个merge layer，会对图片做一个下采样，所以这里要除以2
            bbox_mask = torch.zeros_like(image_mask, dtype=torch.bool)

            if bbox is not None:
                # 从bbox归一化坐标映射到grid上
                x1 = max(0, min((bbox[0] * w).floor().int(), w))
                y1 = max(0, min((bbox[1] * h).floor().int(), h))
                x2 = max(0, min((bbox[2] * w + 1).floor().int(), w))
                y2 = max(0, min((bbox[3] * h + 1).floor().int(), h))
                mask_2d = torch.zeros((h, w), dtype=torch.bool) # 创建出一个形状为(h, w)的bool型张量，并初始化为0，即所有元素均为False，后续只需在图像token对应的位置上改为1(True)
                if x2 > x1 and y2 > y1:
                    mask_2d[y1: y2, x1: x2] = 1 # 将bbox区域覆盖到的格子grid设为1(True)
                mask = mask_2d.flatten() # 把二维网格展平成与图像token顺序一致的一维数组
                image_token_pos = torch.where(image_mask)[0] # 因为image_mask是一维，比如 [False, True, True, False, ...]，所以用torch.where(image_mask)会返回只有一个元素的Tuple，形如(tensor([1, 2, ...]),)，所以torch.where(image_mask)[0]就可以获取所有为True的位置索引
                bbox_mask[image_token_pos] = mask # 把bbox区域对应的图像token位置设为True，其余图像token和文本token位置仍未False
            
            bbox_masks.append(bbox_mask)
        bbox_masks = torch.stack(bbox_masks, dim=0)

        # Process queries.
        if all(q is None for q in texts_query):
            batch_query = None
        elif any(q is None for q in texts_query):
            raise ValueError("Some queries are None. This collator does not support None queries yet.")
        else:
            batch_query = self.processor.process_queries(
                queries=cast(List[str], texts_query),
                max_length=self.max_length,
            )

        # Prefix keys to avoid collisions. 
        # batch_doc、batch_query、batch_neg_doc 可能都含相似键（如 input_ids、attention_mask 等），用 prefix_keys(x, "doc_") 把 batch_doc 的所有键前面加上 "doc_" 前缀；其它两块分别加 "query_"、"neg_doc_"
        batch_all = prefix_keys(batch_doc, "doc_")
        if batch_query:
            batch_all.update(prefix_keys(batch_query, "query_"))
        if batch_neg_doc:
            batch_all.update(prefix_keys(batch_neg_doc, "neg_doc_"))
        
        batch_all["doc_bbox_masks"] = bbox_masks # 把刚才算的 bbox_masks 放进去，命名为 "doc_bbox_masks"（意味着它对应 doc_ 侧的图像 tokens）

        return batch_all
