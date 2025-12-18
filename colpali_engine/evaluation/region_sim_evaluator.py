import os
import json
import torch
import heapq
import pytrec_eval
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
import datasets
from datasets import load_dataset
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Tuple, Dict, Union, Callable, Optional
from torch.utils.data import Dataset, DataLoader
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from colpali_engine.utils.torch_utils import get_torch_device

import io
import pandas as pd

class RegionSimEvaluator:
    def __init__(
        self, model_name, dataset_name, dataset_path, batch_size,
        image_inference_path, text_inference_path,
        force_inference, is_visualization
        # infer_bbox, bbox_score_path, bbox_score_method,
        # bbox_threshold, bbox_neighbor_range, bbox_num_process,
        # eval_iou_threshold
    ) -> None:
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        # self.topks = topks
        self.batch_size = batch_size
        self.image_inference_path = image_inference_path
        self.text_inference_path = text_inference_path
        # self.retrieval_reusults_path = retrieval_results_path
        # os.makedirs(os.path.dirname(self.retrieval_reusults_path), exist_ok=True)
        self.force_inference = force_inference
        self.is_visualization = is_visualization
        # self.infer_bbox = infer_bbox
        # self.bbox_score_path = bbox_score_path
        # self.bbox_score_method = bbox_score_method
        # self.bbox_threshold = bbox_threshold
        # self.bbox_neighbor_range = bbox_neighbor_range
        # self.bbox_num_process = bbox_num_process
        # self.eval_iou_threshold = eval_iou_threshold
        self.device = get_torch_device("auto")
        self.model, self.processor = self.build_model_and_processor()
        self.qrels_bbox = None
        if self.dataset_name in ['mpdocvqa', 'arxivqa', 'chartqa', 'infovqa', 'plotqa', 'slidevqa']:
            self.image_dataloader, self.text_dataloader, self.qrels = self.build_visrag_dataloader()
        elif self.dataset_name in ['visualcot_docvqa', 'visualcot_infovqa']:
            self.image_dataloader, self.text_dataloader, self.qrels, self.qrels_bbox = self.build_visualcot_dataloader()

    @property
    def num_images(self):
        return len(self.image_dataloader.dataset)
    
    @property
    def num_texts(self):
        return len(self.text_dataloader.dataset)

    def build_model_and_processor(self):
        model = ColQwen2_5.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            mask_non_image_embeddings=True,
        ).eval()

        processor = ColQwen2_5_Processor.from_pretrained(self.model_name)

        return model, processor

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

    def build_visualcot_dataloader(self):

        class CoTImageDataset(Dataset):
            def __init__(self, dataset_path, dataset_name):
                super().__init__()
                if dataset_name == 'visualcot_docvqa':
                    anno_file = os.path.join(dataset_path, 'viscot_benchmark/benchmark_det/docvqa.jsonl')
                elif dataset_name == 'visualcot_infovqa':
                    anno_file = os.path.join(dataset_path, 'viscot_benchmark/benchmark_det/infographicsvqa.jsonl')
                # elif dataset_name == 'vsr':
                #     anno_file = os.path.join(dataset_path, 'viscot_benchmark/benchmark_det/vsr.jsonl')

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
                elif dataset_name == 'visualcot_infovqa':
                    anno_file = os.path.join(dataset_path, 'viscot_benchmark/benchmark_det/infographicsvqa.jsonl')
                
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
                image_mask = batch_image['input_ids'] == self.model.config.image_token_id
                image_grids.append(batch_image['image_grid_thw'])

                batch_image = batch_image.to(self.device)
                # attention_mask = batch_image['attention_mask']
                with torch.inference_mode():
                    image_embedding = self.model(**batch_image)
                image_ids.extend(batch_image_id)
                image_sizes.extend(batch_image_sizes)
                
                image_embedding = list(torch.split(image_embedding[image_mask.bool()], image_mask.sum(-1).tolist()))
                image_embeddings.extend([e.to("cpu") for e in image_embedding])
            
            image_grids = torch.cat(image_grids, dim=0)
            image_grids = torch.maximum(torch.tensor(1), image_grids // self.processor.image_processor.merge_size)
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

        if self.is_visualization:
            id2image = {}
            # pure_image_dataloader = DataLoader(
            #     self.image_dataloader.dataset,
            #     batch_size=1,
            #     shuffle=False,
            #     num_workers=8,
            #     collate_fn=lambda x: x[0]
            # )
            image_dataset = self.image_dataloader.dataset
            for obj in tqdm(image_dataset):
                if isinstance(image_dataset, datasets.Dataset):
                    image_id = obj["corpus-id"]
                    image = obj["image"]
                else:
                    image_id, image = obj
                id2image[image_id] = image
            
            return

        global_sims = []
        local_sims = []
        for text_id, gt_info in self.qrels_bbox.items():
            gt_image_id = next(iter(gt_info))
            gt_bbox = gt_info[gt_image_id]

            query_idx = query_ids.index(text_id)
            image_idx = image_ids.index(gt_image_id)
            image_sim = scores[query_idx, image_idx]
            image_mask = p_mask[image_idx]
            h, w = image_sizes[image_idx]
            _, grid_h, grid_w = image_grids[image_idx].numpy()
            x1, y1 = np.round(gt_bbox[0]*grid_w/w), np.round(gt_bbox[1]*grid_h/h),
            x2, y2 = np.round(gt_bbox[2]*grid_w/w), np.round(gt_bbox[3]*grid_h/h)
            x1 = int(max(x1, 0))
            x2 = int(min(x2, grid_w))
            y1 = int(max(y1, 0))
            y2 = int(min(y2, grid_h))
            if x2 == x1 and x2 == grid_w:
                x1 = max(x1 - 1, 0)
            elif x2 == x1:
                x2 += 1
            if y2 == y1 and y2 == grid_h:
                y1 = max(y1 - 1, 0)
            elif y2 == y1:
                y2 += 1

            image_sim = image_sim[image_mask].reshape(grid_h, grid_w)
            if x1 >= x2 or y1 >= y2:
                bbox_region_sim = image_sim.mean()
            else:
                bbox_region_sim = image_sim[y1:y2, x1:x2].mean()
            # if bbox_region_sim.isnan():
            #     print()
            global_sims.append(image_sim.mean())
            local_sims.append(bbox_region_sim)

        print('image sim: ', torch.stack(global_sims).mean())
        print('bbox region sim: ', torch.stack(local_sims).mean())

    def run(self):
        # enable_inference = True
        # if os.path.isfile(self.retrieval_reusults_path):
        #     enable_inference = False
        #     with open(self.retrieval_reusults_path, 'r') as f:
        #         retrieval_results = json.load(f)
        #     if len(retrieval_results) != self.num_texts or self.force_inference:
        #         enable_inference = True
        
        # if enable_inference:
        retrieval_results = self.get_bbox_retrieval_results()
        
        # if self.infer_bbox:
        #     retrieval_results_wo_bbox = {}
        #     for qid, res in retrieval_results.items():
        #         res_wo_bbox = {}
        #         for r in res:
        #             if r['image_id'] in res_wo_bbox:
        #                 continue
        #             res_wo_bbox[r['image_id']] = r['score']
        #         retrieval_results_wo_bbox[qid] = res_wo_bbox
        #     return self.eval_image_results(retrieval_results_wo_bbox)
        #     # self.eval_image_bbox_results(retrieval_results)
        # else:
        #     return self.eval_image_results(retrieval_results)