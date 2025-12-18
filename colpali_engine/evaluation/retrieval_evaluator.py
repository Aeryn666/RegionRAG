import os
import json
import torch
import heapq
import pytrec_eval
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from datasets import load_dataset
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Callable, Optional
from torch.utils.data import Dataset, DataLoader
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.models.paligemma.colpali import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device

import io
import pandas as pd

class RetrievalEvaluator:
    def __init__(
        self, model_name, dataset_name, dataset_path, topks, batch_size,
        image_inference_path, text_inference_path,
        retrieval_results_path, force_inference,
        infer_bbox, bbox_score_path, bbox_score_method,
        bbox_threshold, bbox_neighbor_range, bbox_num_process,
        eval_iou_threshold
    ) -> None:
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.topks = topks
        self.batch_size = batch_size
        self.image_inference_path = image_inference_path
        self.text_inference_path = text_inference_path
        self.retrieval_reusults_path = retrieval_results_path
        os.makedirs(os.path.dirname(self.retrieval_reusults_path), exist_ok=True)
        self.force_inference = force_inference
        self.infer_bbox = infer_bbox
        self.bbox_score_path = bbox_score_path
        self.bbox_score_method = bbox_score_method
        self.bbox_threshold = bbox_threshold
        self.bbox_neighbor_range = bbox_neighbor_range
        self.bbox_num_process = bbox_num_process
        self.eval_iou_threshold = eval_iou_threshold
        self.device = get_torch_device("auto")
        self.model, self.processor = self.build_model_and_processor()
        self.qrels_bbox = None
        if self.dataset_name in ['mpdocvqa', 'arxivqa', 'chartqa', 'infovqa', 'plotqa', 'slidevqa']:
            self.image_dataloader, self.text_dataloader, self.qrels = self.build_visrag_dataloader()
        elif self.dataset_name in ['visualcot_docvqa', 'visualcot_infovqa']:
            self.image_dataloader, self.text_dataloader, self.qrels, self.qrels_bbox = self.build_visualcot_dataloader()
        elif self.dataset_name in ['vidore_docvqa', 'vidore_infovqa', 'vidore_arxivqa', 'vidore_shift', 'vidore_ai', 'vidore_energy', 'vidore_government', 'vidore_health', 'vidore_tabfquad', 'vidore_tatdqa']:
            self.image_dataloader, self.text_dataloader, self.qrels = self.build_vidore_dataloader()
        elif self.dataset_name == 'longdoc':  # LongDocURL contains long PDFs with over 20 pages per document.
            self.image_dataloader, self.text_dataloader, self.qrels = self.build_longdoc_dataloader()
        
    @property
    def num_images(self):
        return len(self.image_dataloader.dataset)
    
    @property
    def num_texts(self):
        return len(self.text_dataloader.dataset)

    def build_model_and_processor(self):
        model_name = self.model_name.lower()

        if "colqwen" or "RegionRet" in model_name:
            print(f"Loading ColQwen2.5-VL model from {self.model_name}")
            model = ColQwen2_5.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
                mask_non_image_embeddings=True,
            ).eval()
            processor = ColQwen2_5_Processor.from_pretrained(self.model_name)

        if "colpali" in model_name:
            print(f"Loading ColPali model from {self.model_name}")
            model = ColPali.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            ).eval()
            processor = ColPaliProcessor.from_pretrained(self.model_name)

        else:
            print(f"Loading ColQwen2.5-VL model from {self.model_name}")
            model = ColQwen2_5.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
                mask_non_image_embeddings=True,
            ).eval()
            processor = ColQwen2_5_Processor.from_pretrained(self.model_name)

        return model, processor
    
    def build_longdoc_dataloader(self):

        class LongImageDataset(Dataset):
            def __init__(self, dataset_path):
                super().__init__()
                anno_file = os.path.join(dataset_path, 'LongDocURL_public.jsonl')
                image_ids = []
                with open(anno_file, 'r') as f:
                    for line in f:
                        item = json.loads(line.strip())
                        pdf_id = item['doc_no']
                        start, end = item['start_end_idx']
                        for idx in range(start - 1, end):
                            img_name = f'{pdf_id}_{idx}.png'
                            image_ids.append(img_name)

                self.image_ids = list(set(image_ids))
                self.image_root = os.path.join(dataset_path, 'pdf_pngs/4000-4999')

            def __len__(self):
                return len(self.image_ids)
            
            def __getitem__(self, idx):
                img_id = self.image_ids[idx]
                img_path = os.path.join(self.image_root, img_id[:4], img_id)
                img = Image.open(img_path)
                return img_id, img
            
        @dataclass
        class ImageDataCollator(object):
            """Colaate examples for supervised fine-tuning."""

            processor: Optional[Callable]

            def __call__(self, instances): # [(img_id, img), (img_id, img)]
                image_ids = [i[0] for i in instances]
                images = [i[1] for i in instances]
                image_sizes = [(img.height, img.width) for img in images]
                images = self.processor.process_images(images=images)
                return image_ids, images, image_sizes
            
        class LongQueryDataset(Dataset):
            def __init__(self, dataset_path):
                super().__init__()
                anno_file = os.path.join(dataset_path, 'LongDocURL_public.jsonl')
                with open(anno_file, 'r') as f:
                    data = [json.loads(line.strip()) for line in f]
                self.query_data = []
                self.qrels = {}
                for item in data:
                    pdf_id = item['doc_no']
                    qid = item['question_id']
                    query = item['question']
                    evidence_pages = item['evidence_pages']
                    gt_image_ids = [f'{pdf_id}_{idx-1}.png' for idx in evidence_pages]
                    self.query_data.append([qid, query])
                    self.qrels[qid] = {img_id: 1 for img_id in gt_image_ids}

            def __len__(self):
                return len(self.query_data)

            def __getitem__(self, index):
                qid, query = self.query_data[index]
                return qid, query
            
        @dataclass
        class QueryDataCollator(object):
            """Collate examples for supervised fine-tuning."""

            processor: Optional[Callable]

            def __call__(self, instances):
                query_ids = [i[0] for i in instances]
                queries = [i[1] for i in instances]
                queries = self.processor.process_queries(queries=queries)
                return query_ids, queries
            
        image_dataset = LongImageDataset(self.dataset_path)
        image_dataloader = DataLoader(
            image_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=ImageDataCollator(self.processor),
        )
        
        query_dataset = LongQueryDataset(self.dataset_path)
        query_dataloader = DataLoader(
            query_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=QueryDataCollator(self.processor),
        )
        qrels = query_dataset.qrels

        return image_dataloader, query_dataloader, qrels


    def build_visualcot_dataloader(self):

        class CoTImageDataset(Dataset):
            def __init__(self, dataset_path, dataset_name):
                super().__init__()
                if dataset_name == 'visualcot_docvqa':
                    anno_file = os.path.join(dataset_path, 'viscot_benchmark/benchmark_det/docvqa.jsonl')

                with open(anno_file, 'r') as f:
                    data = [json.loads(line.strip()) for line in f]
                self.image_ids = list(set([item['img_path'] for item in data]))
                self.image_root = dataset_path

            def __len__(self):
                return len(self.image_ids)
            
            def __getitem__(self, idx):
                img_id = self.image_ids[idx]
                img_path = os.path.join(self.image_root, img_id)
                img = Image.open(img_path)
                return img_id, img
            
        @dataclass
        class ImageDataCollator(object):
            """Colaate examples for supervised fine-tuning."""

            processor: Optional[Callable]

            def __call__(self, instances): # [(img_id, img), (img_id, img)]
                image_ids = [i[0] for i in instances]
                images = [i[1] for i in instances]
                image_sizes = [(img.height, img.width) for img in images]
                images = self.processor.process_images(images=images)
                return image_ids, images, image_sizes
            
        class CotQueryDataset(Dataset):
            def __init__(self, dataset_path, dataset_name):
                super().__init__()
                if dataset_name == 'visualcot_docvqa':
                    anno_file = os.path.join(dataset_path, 'viscot_benchmark/benchmark_det/docvqa.jsonl')

                with open(anno_file, 'r') as f:
                    data = [json.loads(line.strip()) for line in f]
                self.query_data = []
                self.qrels = {}
                self.qrels_bbox = {}
                for item in data:
                    qid = str(item['question_id'])
                    query = item['expression']
                    image_id = item["img_path"]
                    # assert len(image_id) == 1
                    # image_id = image_id[0]
                    self.query_data.append([qid, query])
                    self.qrels[qid] = {image_id: 1}
                    self.qrels_bbox[qid] = {image_id: item['bbox']}

            def __len__(self):
                return len(self.query_data)

            def __getitem__(self, index):
                qid, query = self.query_data[index]
                return qid, query

        
        @dataclass
        class QueryDataCollator(object):

            processor: Optional[Callable]

            def __call__(self, instances):
                query_ids = [i[0] for i in instances]
                queries = [i[1] for i in instances]
                queries = self.processor.process_queries(queries=queries)
                return query_ids, queries
            
        image_dataset = CoTImageDataset(self.dataset_path, self.dataset_name)
        image_dataloader = DataLoader(
            image_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=ImageDataCollator(self.processor),
        )

        query_dataset = CotQueryDataset(self.dataset_path, self.dataset_name)
        query_dataloader = DataLoader(
            query_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=QueryDataCollator(self.processor),
        )

        qrels = query_dataset.qrels
        qrels_bbox = query_dataset.qrels_bbox

        return image_dataloader, query_dataloader, qrels, qrels_bbox


    def build_vidore_dataloader(self):
        
        @dataclass
        class ImageDataCollator(object):
            """Collate examples for supervised fine-tuning."""

            processor: Optional[Callable]

            def __call__(self, instances):
                image_ids, images = tuple([instance[key] for instance in instances] for key in ("image_filename", "image"))
                image_sizes = [(img.height, img.width) for img in images]
                images = self.processor.process_images(images=images)
                return image_ids, images, image_sizes

        @dataclass
        class QueryDataCollator(object):
            """Collate examples for supervised fine-tuning."""

            processor: Optional[Callable]

            def __call__(self, instances):
                query_ids, queries = tuple([instance[key] for instance in instances] for key in ("questionId", "query"))
                queries = self.processor.process_queries(queries=queries)
                return query_ids, queries
            

        dataset = load_dataset(self.dataset_path, split="test")
        dataset = dataset.filter(lambda x: x["query"] is not None)
        if "questionId" not in dataset.column_names:
            query_ids = [str(i) for i in range(len(dataset))]
            dataset = dataset.add_column("questionId", query_ids)
        dataset_no_image = dataset.remove_columns(["image"])

        image_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=ImageDataCollator(self.processor),
        )

        # query_dataset = load_dataset(self.dataset_path, 'data', split="test")
        query_dataloader = DataLoader(
            dataset_no_image,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=QueryDataCollator(self.processor),
        )

        qrels = {}

        qids = dataset['questionId']
        img_ids = dataset['image_filename']
        qrels = {qid: {img_id: 1} for qid, img_id in zip(qids, img_ids)}

        return image_dataloader, query_dataloader, qrels

    def build_visrag_dataloader(self):

        @dataclass 
        class CorpusDataCollator(object): 
            """Collate examples for supervised fine-tuning."""

            processor: Optional[Callable]

            def __call__(self, instances): 
                corpus_ids, images = tuple([instance[key] for instance in instances] for key in ("corpus-id", "image"))
                image_sizes = [(img.height, img.width) for img in images]
                images = self.processor.process_images(images=images)
                return corpus_ids, images, image_sizes


        @dataclass
        class QueryDataCollator(object):
            """Collate examples for supervised fine-tuning."""

            processor: Optional[Callable]

            def __call__(self, instances): 
                query_ids, queries = tuple([instance[key] for instance in instances] for key in ("query-id", "query"))
                queries = self.processor.process_queries(queries=queries)
                return query_ids, queries


        corpus_dataset = load_dataset(self.dataset_path, 'corpus', split="train")
        corpus_dataloader = DataLoader(
            corpus_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=CorpusDataCollator(self.processor),
        ) 

        query_dataset = load_dataset(self.dataset_path, 'queries', split="train")


        query_dataloader = DataLoader(
            query_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=QueryDataCollator(self.processor),
        )

        qrels_ds = load_dataset(self.dataset_path, 'qrels', split="train")
        qrels = {q['query-id']: {q['corpus-id']: q['score']} for q in qrels_ds}

        return corpus_dataloader, query_dataloader, qrels

    def inference_image(self):
        image_ids = []
        image_embeddings = []
        image_sizes = []
        image_grids = []

        need_inference = True
        if os.path.isfile(self.image_inference_path):
            saved_image_features = torch.load(self.image_inference_path)
            if len(saved_image_features['image_id']) != self.num_images or self.force_inference:
                need_inference = True
            else:
                need_inference = False
                image_ids = saved_image_features['image_id']
                image_embeddings = saved_image_features['image_embedding']
                image_sizes = saved_image_features['image_size']
                image_grids = saved_image_features['image_grid']
        
        if need_inference:
            for batch_image_id, batch_image, batch_image_sizes in tqdm(self.image_dataloader, total=len(self.image_dataloader)):
                
                if hasattr(self.model.config, "image_token_id"):
                    image_mask = batch_image['input_ids'] == self.model.config.image_token_id
                elif hasattr(self.model.config, "image_token_index"):
                    image_mask = batch_image['input_ids'] == self.model.config.image_token_index
                else:
                    image_mask = None  

                if "image_grid_thw" in batch_image:
                    grids = batch_image['image_grid_thw']
                else:
                    grids = []
                    for pixel_value in batch_image['pixel_values']:
                        _, H, W = pixel_value.shape
                        h = H // 14 
                        w = W // 14 
                        t = 1
                        grids.append((t, h, w))
                    grids = torch.tensor(grids)
                image_grids.append(grids)

                batch_image = batch_image.to(self.device)
                # attention_mask = batch_image['attention_mask']
                with torch.inference_mode():
                    image_embedding = self.model(**batch_image)
                image_ids.extend(batch_image_id)
                image_sizes.extend(batch_image_sizes)
                
                image_embedding = list(torch.split(image_embedding[image_mask.bool()], image_mask.sum(-1).tolist()))
                image_embeddings.extend([e.to("cpu") for e in image_embedding])
            
            image_grids = torch.cat(image_grids, dim=0)
            if isinstance(self.processor, ColQwen2_5_Processor):
                image_grids = torch.maximum(torch.tensor(1), image_grids // self.processor.image_processor.merge_size)
            else:
                image_grids = torch.maximum(torch.tensor(1), image_grids)

            if self.image_inference_path is not None:
                torch.save(
                    {
                        "image_id": image_ids,
                        "image_embedding": image_embeddings,
                        "image_size": image_sizes,
                        "image_grid": image_grids,
                    },
                    # {k: v for k,v in zip(image_ids, image_embeddings)},
                    self.image_inference_path
                )
        
        return image_ids, image_embeddings, image_sizes, image_grids

    def inference_text(self):
        text_ids = []
        text_embeddings = []

        need_inference = True
        if os.path.isfile(self.text_inference_path):
            text_features = torch.load(self.text_inference_path)
            if len(text_features) != self.num_texts or self.force_inference:
                need_inference = True
            else:
                need_inference = False
                for k, v in text_features.items():
                    text_ids.append(k)
                    text_embeddings.append(v)
        
        if need_inference:
            for batch_text_id, batch_text in tqdm(self.text_dataloader, total=len(self.text_dataloader)):
                batch_text = batch_text.to(self.device)
                attention_mask = batch_text['attention_mask']
                with torch.inference_mode():
                    text_embedding = self.model(**batch_text)
                text_ids.extend(batch_text_id)
                text_embedding = list(torch.split(text_embedding[attention_mask.bool()], attention_mask.sum(-1).tolist()))
                text_embeddings.extend([e.to("cpu") for e in text_embedding])
            if self.text_inference_path is not None:
                torch.save(
                    {k: v for k,v in zip(text_ids, text_embeddings)},
                    self.text_inference_path
                )
        
        return text_ids, text_embeddings

    def get_retrieval_results(self):
        image_ids, image_embeddings, image_sizes, image_grids = self.inference_image()
        query_ids, query_embeddings = self.inference_text()
        # Compute retrieval scores
        # scores = self.processor.score_multi_vector(
        #     qs=query_embeddings,
        #     ps=image_embeddings,
        # )  # (len(qs), len(ps))
        scores, p_mask = self.processor.score_multi_vector_per_patch(
            qs=query_embeddings,
            ps=image_embeddings
        )
        scores = scores.max(dim=-1).values

        max_topk = max(self.topks)
        topk_scores, topk_indices = torch.topk(scores, max_topk, dim=1)

        retrieval_results = {}
        for qid in query_ids:
            retrieval_results[qid] = {}
        
        for q in range(topk_scores.shape[0]):
            qid = query_ids[q]
            for idx, score in zip(topk_indices[q], topk_scores[q]):
                vid = image_ids[idx.item()]
                retrieval_results[qid][vid] = score.item()

        if self.retrieval_reusults_path is not None:
            with open(self.retrieval_reusults_path, 'w') as f:
                json.dump(retrieval_results, f, ensure_ascii=False, indent=4)
        
        return retrieval_results

    def get_bbox_retrieval_results(self):
        image_ids, image_embeddings, image_sizes, image_grids = self.inference_image()
        query_ids, query_embeddings = self.inference_text()

        # Compute retrieval scores
        n_queries = len(query_ids)
        n_images = len(image_ids)

        scores, p_mask = self.processor.score_multi_vector_per_patch(
            qs=query_embeddings,
            ps=image_embeddings
        )

        grid_sizes = torch.tensor(image_sizes) / image_grids[:, 1:]

        results = []

        def generate_task_args():
            for q_idx in range(n_queries):
                for p_idx in range(n_images):
                    # Scores for this query-image pair
                    query_id = query_ids[q_idx]
                    image_id = image_ids[p_idx]
                    score = scores[q_idx, p_idx][p_mask[p_idx]] # Shape: (max_patches_per_image,)
                    image_size = image_sizes[p_idx]
                    image_grid = image_grids[p_idx].tolist()
                    grid_size = grid_sizes[p_idx].tolist()
                    yield (query_id, image_id, score, self.bbox_score_method, self.bbox_threshold, 
                           self.bbox_neighbor_range, image_size, image_grid, grid_size)

        if self.bbox_num_process == 0:
            for arg in generate_task_args():
                single_result = self.processor.single_get_box(arg)
                results.extend(single_result)
        else:
            with Pool(processes=self.bbox_num_process) as pool:
                result = list(tqdm(pool.imap_unordered(self.processor.single_get_box, generate_task_args()), total=n_queries*n_images))
                # results = list(pool.imap_unordered(self.processor.single_get_box, task_args))
            for res in result:
                results.extend(res)
        
        retrieval_results = defaultdict(list)
        for res in results:
            query_id = res.pop("query_id")
            retrieval_results[query_id].append(res)
        
        # Sort regions for the current query by score (descending)
        max_topk = max(self.topks)
        # for qid in retrieval_results.keys():
        #     # retrieval_results[qid].sort(key=lambda x: x["score"], reverse=True)
        #     retrieval_results[qid] = heapq.nlargest(max_topk, retrieval_results[qid], key=lambda x: x["score"])
        retrieval_results = self.parallel_get_top_k_retrieval_results(retrieval_results, max_topk)


        if self.retrieval_reusults_path is not None:
            with open(self.retrieval_reusults_path, 'w') as f:
                json.dump(retrieval_results, f, ensure_ascii=False, indent=4)

        return retrieval_results

    @staticmethod
    def get_top_k_results(retrieval_result_for_qid, k):
        qid, results = retrieval_result_for_qid
        top_k_results = heapq.nlargest(k, results, key=lambda x: x["score"])
        return qid, top_k_results

    @staticmethod
    def parallel_get_top_k_retrieval_results(retrieval_results, k):
        with Pool(processes=20) as pool:
            top_k_results = pool.starmap(RetrievalEvaluator.get_top_k_results, [(item, k) for item in retrieval_results.items()])
        top_k_retrieval_results = {qid: results for qid, results in top_k_results}
        return top_k_retrieval_results
    
    def eval_image_results(self, retrieval_results):
        # max_topk = min([len(v) for k, v in retrieval_results.items()])
        # topks = [min(max_topk, k) for k in self.topks]
        topks = self.topks

        metrics = {f"ndcg_cut.{topk}" for topk in topks}
        metrics.update({f"recall.{topk}" for topk in topks})
        evaluator = pytrec_eval.RelevanceEvaluator(self.qrels, metrics)
        eval_results = evaluator.evaluate(retrieval_results)

        query_id, query_measures = sorted(eval_results.items())[-1]

        results = {}
        # for measure in sorted(query_measures.keys()):
        #     results[measure] = pytrec_eval.compute_aggregated_measure(
        #         measure, [query_measures[measure] for query_measures in eval_results.values()]
        #     )
        for topk in topks:
            results[f"recall@{topk}"] = pytrec_eval.compute_aggregated_measure(
                f"recall_{topk}", [query_measures[f"recall_{topk}"] for query_measures in eval_results.values()]
            )
        for topk in topks:
            results[f"ndcg@{topk}"] = pytrec_eval.compute_aggregated_measure(
                f"ndcg_cut_{topk}", [query_measures[f"ndcg_cut_{topk}"] for query_measures in eval_results.values()]
            )
        for topk in topks:
            results[f'mrr@{topk}'] = self.eval_mrr(self.qrels, retrieval_results, topk)['all']

        for k, v in results.items():
            results[k] = v * 100
            print("{:20s}{:.4f}".format(k, results[k]))
        
        return results

    def calculate_iou_batch(self, pred_boxes: np.ndarray, gt_box: np.ndarray) -> np.ndarray:
        """
        使用 NumPy 高效地批量计算一组预测框与单个真实框之间的 IoU。

        Args:
            pred_boxes (np.ndarray): 预测的边界框数组，形状为 (N, 4)，N是预测框的数量。
            gt_box (np.ndarray): 单个真实的边界框数组，形状为 (4,)。

        Returns:
            np.ndarray: 一个包含 N 个 IoU 值的一维数组。
        """
        # 确保输入是 numpy array
        pred_boxes = np.asarray(pred_boxes)
        gt_box = np.asarray(gt_box)

        # 1. 计算交集区域的坐标 (x1, y1, x2, y2)
        # np.maximum 逐元素比较，找到交集左上角的最大坐标
        inter_x1 = np.maximum(pred_boxes[:, 0], gt_box[0])
        inter_y1 = np.maximum(pred_boxes[:, 1], gt_box[1])
        # np.minimum 逐元素比较，找到交集右下角的最小坐标
        inter_x2 = np.minimum(pred_boxes[:, 2], gt_box[2])
        inter_y2 = np.minimum(pred_boxes[:, 3], gt_box[3])

        # 2. 计算交集区域的面积
        # np.maximum(0, ...) 确保在没有重叠时面积为0，而不是负数
        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

        # 3. 计算各个框的面积
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

        # 4. 计算并集面积
        union_area = pred_area + gt_area - inter_area

        # 5. 计算 IoU
        # 加上一个极小值 epsilon 防止除以0
        iou = inter_area / (union_area + 1e-6)

        return iou

    def eval_image_bbox_results(self, retrieval_results,) -> Dict[str, float]:
        """
        为 Visual Grounding 任务计算 Top-K 准确率。

        Args:
            retrieval_results (Dict): 模型的预测结果。
                                    key是text_id, value是按score排序的预测列表。
            qrels (Dict): 真实的标注数据 (Ground Truth)。
                        key是text_id, value是 {image_id: bounding_box}。
            iou_threshold (float): 判断预测是否正确的 IoU 阈值。
            top_k_values (Union[List[int], Tuple[int, ...]]): 需要计算的 K 值列表。

        Returns:
            Dict[str, float]: 一个字典，key是 'Top-K Acc'，value 是对应的准确率。
        """
        # 初始化命中计数器
        hits = {k: 0 for k in self.topks}
        total_queries = len(self.qrels_bbox)
        max_k = max(self.topks)

        # 遍历每一个有真实标注的查询
        for text_id, gt_info in self.qrels_bbox.items():
            # 如果预测结果中没有这个 text_id，则直接跳过（视为未命中）
            text_id = str(text_id)
            if text_id not in retrieval_results or not retrieval_results[text_id]:
                continue

            # 提取真实标注信息
            # gt_info 是一个字典，但通常只有一个键值对
            try:
                gt_image_id = next(iter(gt_info))
                gt_bbox = np.array(gt_info[gt_image_id])
            except StopIteration:
                continue # 如果 gt_info 为空，则跳过

            # 获取 Top-K 预测结果
            predictions = retrieval_results[text_id][:max_k]
            if not predictions:
                continue

            # 提取与 gt_image_id 匹配的预测框
            matched_pred_indices = []
            matched_pred_boxes = []
            for i, pred in enumerate(predictions):
                if pred['image_id'] == gt_image_id:
                    matched_pred_indices.append(i)
                    matched_pred_boxes.append(pred['bounding_box'])
            
            if not matched_pred_boxes:
                continue # 如果 top-K 预测中没有一个在正确的图片上，则未命中

            # 批量计算 IoU
            ious = self.calculate_iou_batch(np.array(matched_pred_boxes), gt_bbox)

            # 检查在 Top-K 中是否有命中
            # 找到第一个命中的预测在原始 top-k 列表中的索引（0-based）
            first_hit_index = -1
            for i, idx in enumerate(matched_pred_indices):
                if ious[i] >= self.eval_iou_threshold:
                    first_hit_index = idx
                    break  # 只要找到第一个命中的就可以停止

            # 如果找到了命中，则更新所有相关的 K 值的命中数
            if first_hit_index != -1:
                for k in self.topks:
                    # 如果第一个命中的索引小于 k，说明它在 Top-K 范围内
                    if first_hit_index < k:
                        hits[k] += 1
        
        # 计算最终的准确率
        accuracies = {f'Top-{k} Acc': hits[k] / total_queries for k in self.topks}
        print(accuracies)
        return accuracies



    @staticmethod
    def eval_mrr(qrel, run, cutoff=None):
        """
        Compute MRR@cutoff manually.
        """
        mrr = 0.0
        num_ranked_q = 0
        results = {}
        for qid in qrel:
            if qid not in run:
                continue
            num_ranked_q += 1
            docid_and_score = [(docid, score) for docid, score in run[qid].items()]
            docid_and_score.sort(key=lambda x: x[1], reverse=True)
            for i, (docid, _) in enumerate(docid_and_score):
                rr = 0.0
                if cutoff is None or i < cutoff:
                    if docid in qrel[qid] and qrel[qid][docid] > 0:
                        rr = 1.0 / (i + 1)
                        break
            results[qid] = rr
            mrr += rr
        mrr /= num_ranked_q
        results["all"] = mrr
        return results

    def run(self):
        enable_inference = True
        if os.path.isfile(self.retrieval_reusults_path):
            enable_inference = False
            with open(self.retrieval_reusults_path, 'r') as f:
                retrieval_results = json.load(f)
            if len(retrieval_results) != self.num_texts or self.force_inference:
                enable_inference = True
        
        if enable_inference:
            retrieval_results = self.get_bbox_retrieval_results() if self.infer_bbox else self.get_retrieval_results()
        
        if self.infer_bbox:
            retrieval_results_wo_bbox = {}
            for qid, res in retrieval_results.items():
                res_wo_bbox = {}
                for r in res:
                    if r['image_id'] in res_wo_bbox:
                        continue
                    res_wo_bbox[r['image_id']] = r['score']
                retrieval_results_wo_bbox[qid] = res_wo_bbox
            return self.eval_image_results(retrieval_results_wo_bbox)
            # self.eval_image_bbox_results(retrieval_results)
        else:
            return self.eval_image_results(retrieval_results)