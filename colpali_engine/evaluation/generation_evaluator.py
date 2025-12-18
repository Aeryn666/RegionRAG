import os
import io
import time
import json
import torch
import base64
import pandas as pd
import random
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from collections import defaultdict
from datasets import load_dataset
from dataclasses import dataclass, field
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from openai import OpenAI
from typing import Callable, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from concurrent.futures import (
    as_completed,
    ThreadPoolExecutor,
)

from .eval_utils import check_responses
from colpali_engine.utils.torch_utils import is_main_process, synchronize


class GenerationEvaluator:
    def __init__(
        self, model_path, deploy_type, max_new_tokens,
        dataset_name, dataset_path, retrieval_results_path,
        force_generate, topk_image, max_image_pixels, min_image_pixels,
        max_bbox_pixels, min_bbox_pixels, batch_size, response_path, infer_visual_type,
        vllm_tensor_parallel_size=1, api_num_processes=1, world_size=1, rank=0
    ) -> None:
        self.model_path = model_path
        self.deploy_type = deploy_type
        self.max_new_tokens = max_new_tokens
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.retrieval_results_path = retrieval_results_path
        self.force_generate = force_generate
        self.topk_image = topk_image
        self.max_image_pixels = max_image_pixels
        self.min_image_pixels = min_image_pixels
        self.max_bbox_pixels = max_bbox_pixels
        self.min_bbox_pixels = min_bbox_pixels
        self.batch_size = batch_size
        self.response_path = response_path
        self.infer_visual_type = infer_visual_type
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.api_num_processes = api_num_processes
        self.world_size = world_size
        self.rank = rank
        self.records = []

        if self.model_path is not None:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            if self.dataset_name in ['mpdocvqa', 'arxivqa', 'chartqa', 'infovqa', 'plotqa', 'slidevqa']:
                self.dataloader = self.build_visrag_dataloader()
            if self.dataset_name in ['vidore_docvqa', 'vidore_infovqa', 'vidore_arxivqa']:
                self.dataloader = self.build_vidore_dataloader()
        
            if self.force_generate:
                if is_main_process():
                    if os.path.isfile(self.response_path):
                        print(f"Warning: force_generate is enabled but response file exists, removing {self.response_path}")
                        os.remove(self.response_path)
            
            if os.path.isfile(self.response_path):
                self.records = self.read_response()
                if len(self.records) != self.dataloader.dataset.total_len:
                    self.records = []
                    if is_main_process():
                        print(f"Warning: Invalid response file exists, removing {self.response_path}")
                        os.remove(self.response_path)
            
            if len(self.records) != self.dataloader.dataset.total_len:
                if self.deploy_type == "transformers":
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                        device_map="auto"
                    )
                    self.sampling_params = dict(
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        temperature=0,
                        top_p=0.001,
                    )
                elif self.deploy_type == "vllm_local":
                    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
                    self.model = LLM(
                        model=self.model_path,
                        limit_mm_per_prompt={"image": 10, "video": 10},
                        tensor_parallel_size=self.vllm_tensor_parallel_size
                    )
                    self.sampling_params = SamplingParams(
                        temperature=0,
                        top_p=0.001,
                        repetition_penalty=1.05,
                        max_tokens=self.max_new_tokens,
                        stop_token_ids=[],
                    )
                elif self.deploy_type == "api":
                    assert self.batch_size == 1
                    self.sampling_params = dict(
                        max_tokens=self.max_new_tokens,
                        # do_sample=False,
                        temperature=0,
                        top_p=0.001,
                    )
    
    def read_response(self):
        with open(self.response_path, 'r') as f:
            records = [json.loads(l.strip()) for l in f.readlines()]
        return records


    def build_visrag_dataloader(self):
        corpus = load_dataset(self.dataset_path, 'corpus', split="train").to_list()
        id2corpus = {c['corpus-id']: c['image']['bytes'] for c in corpus}
        query = load_dataset(self.dataset_path, 'queries', split="train").to_list() 

        if "options" in query[0] and query[0]["options"] is not None:
            id2query = {q['query-id']: {
                'question': q['query'], 'answer': q['answer'], 'options': q['options']
            } for q in query}
        else:
            id2query = {q['query-id']: {'question': q['query'], 'answer': q['answer']} for q in query}
        with open(self.retrieval_results_path, 'r') as f:
            retrieval_results = json.load(f)
        
        if self.infer_visual_type == "oracle":
            qrels = load_dataset(self.dataset_path, 'qrels', split="train").to_list()
            qid2vid = {qrel['query-id']: qrel['corpus-id'] for qrel in qrels}
            for qid, item in retrieval_results.items():
                retrieval_results[qid] = {qid2vid[qid]: 1}

        dataset = QwenDataset(
            id2corpus, id2query, retrieval_results,
            self.infer_visual_type, self.topk_image, self.processor,
            self.deploy_type, data_root=None,
            max_image_pixels=self.max_image_pixels,
            min_image_pixels=self.min_image_pixels,
            max_bbox_pixels=self.max_bbox_pixels,
            min_bbox_pixels=self.min_bbox_pixels,
            num_split=self.world_size, split_idx=self.rank,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=20 if self.deploy_type == "api" else 8,
            collate_fn=DataCollator(self.deploy_type, self.processor),
        )

        return dataloader


    def build_visualcot_dataloader(self):
        # id2corpus = {'image_id(xxx.jpg)': image_bytes}
        # id2query = {'question_id('xxx')': {'question': 'xxx', 'answer': ['xxx']}}
        if self.dataset_name == 'textvqa':
            anno_file = os.path.join(self.dataset_path, 'metadata/textvqa_cot_train.jsonl')
        elif self.dataset_name == 'vsr':
            anno_file = os.path.join(self.dataset_path, 'metadata/vsr_cot_train.jsonl')

        with open(anno_file, 'r') as f:
            data = [json.loads(line.strip()) for line in f]
        id2corpus = {}
        id2query = {}
        for idx, item in enumerate(data, start=1):
            qid = str(idx)
            query = item['question']
            answer = item['answer']
            img_id = item['image']
            img_path = os.path.join(self.dataset_path, 'images', self.dataset_name, img_id)
            img = Image.open(img_path)
            id2corpus[img_id] = img
            id2query[qid] = {'question': query, 'answer': answer}

        with open(self.retrieval_results_path, 'r') as f:
            retrieval_results = json.load(f)
        dataset = QwenDataset(
            id2corpus, id2query, retrieval_results,
            self.topk_image, self.processor, data_root=None,
            max_pixels=self.max_pixels
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=DataCollator(),
        )

        return dataloader    


    def batch_generate(self):
        if self.deploy_type == "api":
            return self.api_generate()
        records = []
        for i, (text_ids, texts, llm_inputs, num_visual_tokens) in enumerate(tqdm(self.dataloader)):
            if self.deploy_type == "transformers":
                llm_inputs = llm_inputs.to("cuda")
                generated_ids = self.model.generate(**llm_inputs, **self.sampling_params)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(llm_inputs.input_ids, generated_ids)
                ]
                generated_texts = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            elif self.deploy_type == "vllm_local":
                outputs = self.model.generate(llm_inputs, sampling_params=self.sampling_params)
                generated_texts = [outputs[j].outputs[0].text for j in range(len(llm_inputs))]
            
            for tid, text, response, num_visual_token in zip(text_ids, texts, generated_texts, num_visual_tokens):
                record = {
                    "question_id": tid,
                    'question': text['question'],
                    'answer': text['answer'],
                    'response': response,
                    'num_visual_token': num_visual_token,
                }
                if 'options' in text:
                    record['options'] = text['options']
                    record['response'] = response.strip(".").lstrip("(").rstrip(")")
                records.append(record)
                if 'options' in text:
                    record['options'] = text['options']
                    record['response'] = response.strip(".").lstrip("(").rstrip(")")
                records.append(record)
        with open(self.response_path, 'a', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        return records

    def curl_func(self, messages):
        try:
            openai_api_key = "EMPTY"
            openai_api_base = "http://localhost:8000/v1"
            client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            chat_response = client.chat.completions.create(
                model="Qwen2.5-VL-7B-Instruct",
                messages=messages,
                **self.sampling_params,
            )

            return chat_response.choices[0].message.content
        except Exception as err:
            print(err)
            return None
        
    def curl_gpt_func(self, messages):
        try:
            openai_api_key = "xxx"  # Replace it with the user’s real OpenAI API key
            openai_api_base = "https://xxx"  # Replace it with the user’s real OpenAI API kbase
            client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
            )
            chat_response = client.chat.completions.create(
                model="o3",  # Can replace it with gpt-5
                messages=messages,
                **self.sampling_params,
            )
            # print(chat_response.choices[0].message.content)
            return chat_response.choices[0].message.content
        except Exception as err:
            err_str = str(err)
            print("Error:", err_str)
            if "content management policy" in err_str or "content_filter" in err_str:
                return "__CONTENT_FILTER__"     
        return "__API_ERROR__"

    def api_worker(self, args):
        text_ids, texts, messages, num_visual_tokens = args
        tid = text_ids[0]
        text = texts[0]
        message = messages[0]
        num_visual_token = num_visual_tokens[0]
        for i in range(5):
            response = self.curl_func(message)  # use qwen api
            # response = self.curl_gpt_func(message)  # use gpt api
            if response in ["__API_ERROR__", "__CONTENT_FILTER__", None]:
                if response == "__CONTENT_FILTER__":
                    print(f"Stopped retry due to content filter at attempt {i+1}")
                    break
                else:
                    time.sleep(3)
                    print(f"Retrying {i+1} times")
            else:
                break
        
        if response in ["__API_ERROR__", "__CONTENT_FILTER__", None]:
            if "options" in text:  
                response = random.choice(text["options"])
            else:  
                response = "0"
        
        
        if response is None:
            return None
        
        record = {
            "question_id": tid,
            'question': text['question'],
            'answer': text['answer'],
            'response': response,
            'num_visual_token': num_visual_token,
        }
        if 'options' in text:
            record['options'] = text['options']
            record['response'] = response.strip(".").lstrip("(").rstrip(")")
        if 'options' in text:
            record['options'] = text['options']
            record['response'] = response.strip(".").lstrip("(").rstrip(")")
        return record

    def api_generate(self):
        results = []
        with ThreadPoolExecutor(max_workers=self.api_num_processes) as executor:
            futures = {executor.submit(self.api_worker, arg): arg for arg in self.dataloader}
            for future in tqdm(as_completed(futures), total=len(self.dataloader), desc=f"worker"):
                result = future.result()
                if result is not None:
                    results.append(result)
            
            with open(self.response_path, 'a', encoding='utf-8') as f:
                for record in results:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

        return results

    def eval_results(self):
        # if records is None:
        #     with open(self.response_path, 'r') as f:
        #         records = [json.loads(l.strip()) for l in f.readlines()]

        correct = 0
        num_visual_tokens = []
        for record in self.records:
            question = record['question']
            answer = record['answer']
            response = record['response']
            num_visual_tokens.append(record['num_visual_token'])

            response_correct, processed_response, processed_answer = check_responses(
                self.dataset_name, response, answer, question
            )
            correct += response_correct

        acc = float(correct) / len(self.records)
        avg_num_visual_tokens = sum(num_visual_tokens) / len(num_visual_tokens)
        print(f"{self.dataset_name}:{len(self.records)}_Accuracy:{acc}")
        print(f"Avg num_visual_tokens: {avg_num_visual_tokens}")
        return {"acc": acc, "avg_num_visual_tokens": avg_num_visual_tokens}

    def run(self):
        if len(self.records) != self.dataloader.dataset.total_len:
            self.batch_generate()
        synchronize()

        if is_main_process():
            self.records = self.read_response()
            results = self.eval_results()
        else:
            results = {}

        synchronize()
        return results



class QwenDataset(Dataset):
    def __init__(
        self, id2image, id2text, retrieval_results,
        infer_visual_type, topk_image, processor,
        deploy_type, data_root=None, 
        max_image_pixels=1024*1024, min_image_pixels=256*256,
        max_bbox_pixels=1024*1024, min_bbox_pixels=256*256,
        num_split=1, split_idx=0,
    ):
        super(QwenDataset, self).__init__()
        self.id2image = id2image
        self.id2text = id2text
        self.text_ids = list(self.id2text.keys())[split_idx::num_split]
        self.retrieval_results = retrieval_results
        self.infer_visual_type = infer_visual_type
        self.topk_image = topk_image
        self.processor = processor
        self.deploy_type = deploy_type
        self.data_root = data_root
        self.max_image_pixels=max_image_pixels
        self.min_image_pixels=min_image_pixels
        self.max_bbox_pixels=max_bbox_pixels
        self.min_bbox_pixels=min_bbox_pixels

        if self.infer_visual_type == "oracle":
            self.topk_image = 1

        if self.infer_visual_type == "oracle":
            self.topk_image = 1

    @property
    def total_len(self):
        return len(self.retrieval_results)
    
    def __len__(self):
        return len(self.text_ids)
    
    def __getitem__(self, index):
        text_id = self.text_ids[index]
        item = self.id2text[text_id]
        question = item['question']
        if "options" in item:
            options_string = "\n".join(item["options"])
            prompt = f'Question: {question}\nOptions:\n{options_string}\nAnswer with the option\'s letter from the given choices directly and only give the best option.'
        else:
            prompt = f"{question} Answer the question using a single word or phrase. Answer:"
        item = self.id2text[text_id]
        question = item['question']
        if "options" in item:
            options_string = "\n".join(item["options"])
            prompt = f'Question: {question}\nOptions:\n{options_string}\nAnswer with the option\'s letter from the given choices directly and only give the best option.'
        else:
            prompt = f"{question} Answer the question using a single word or phrase. Answer:"
        if 'box' in self.infer_visual_type:
            retrieval_results = self.retrieval_results[text_id][:self.topk_image]
            image_ids = [r['image_id'] for r in retrieval_results]
            bounding_boxes = [r['bounding_box'] for r in retrieval_results]
        else:
            if isinstance(self.retrieval_results[text_id], dict):
                image_ids = list(self.retrieval_results[text_id].keys())[:self.topk_image]
            else:
                retrieval_results = self.retrieval_results[text_id][:self.topk_image]
                image_ids = [r['image_id'] for r in retrieval_results]
            if isinstance(self.retrieval_results[text_id], dict):
                image_ids = list(self.retrieval_results[text_id].keys())[:self.topk_image]
            else:
                retrieval_results = self.retrieval_results[text_id][:self.topk_image]
                image_ids = [r['image_id'] for r in retrieval_results]
            bounding_boxes = [[] for i in image_ids]

        id2images = {}
        id2region_images = defaultdict(list)
        for image_id, box in zip(image_ids, bounding_boxes):
            if isinstance(self.id2image[image_id], str):
                # image = f"file://{os.path.join(self.data_root, self.id2image[image_id])}"
                image_path = os.path.join(self.data_root, self.id2image[image_id])
                image = Image.open(image_path)
            elif isinstance(self.id2image[image_id], bytes):
                # image_base64 = base64.b64encode(self.id2image[image_id]).decode('utf-8')
                # image = f"data:image/png;base64,{image_base64}"
                image_bytes = BytesIO(self.id2image[image_id])
                image = Image.open(image_bytes)
            else:
                image = self.id2image[image_id]
            
            if 'box' in self.infer_visual_type:
                region_image = image.crop(box)
                id2region_images[image_id].append((box, region_image))
            
            id2images[image_id] = image
        
        if 'box' in self.infer_visual_type and 'image' in self.infer_visual_type:
            system_prompt = "You are a helpful assistent. You will be given a question and some retrieved images that are likely related to the question. Please anaylze the visual content and answer the question."
            exist_image_id_pool = set()
            content = []
            for image_id in image_ids:
                if image_id in exist_image_id_pool:
                    continue
                exist_image_id_pool.add(image_id)
                for i, (box, region_image) in enumerate(id2region_images[image_id]):
                    content.append({
                        "type": "image", 
                        "image": region_image, 
                        "max_pixels": self.max_bbox_pixels,
                        "min_pixels": self.min_bbox_pixels
                    })
                content.append({
                    "type": "image", 
                    "image": image, 
                    "max_pixels": self.max_image_pixels,
                    "min_pixels": self.min_image_pixels
                })
                

        elif self.infer_visual_type == 'bbox':
            system_prompt = "You are a helpful assistent. You will be given a question and some retrieved images that are likely related to the question. Please anaylze the visual content and answer the question."
            exist_image_id_pool = set()
            content = []
            for image_id in image_ids:
                if image_id in exist_image_id_pool:
                    continue
                exist_image_id_pool.add(image_id)
                for i, (box, region_image) in enumerate(id2region_images[image_id]):
                    content.append({
                        "type": "image",
                        "image": region_image,
                        "max_pixels": self.max_bbox_pixels,
                        "min_pixels": self.min_bbox_pixels
                    })
        elif self.infer_visual_type == 'image':
            system_prompt = "You are a helpful assistent. You will be given a question and some retrieved images that are likely related to the question. Please anaylze the visual content and answer the question."
            content = [{
                "type": "image",
                "image": id2images[image_id],
                "max_pixels": self.max_image_pixels,
                "min_pixels": self.min_image_pixels
            } for image_id in image_ids]
        elif self.infer_visual_type == "oracle":
            system_prompt = "You are a helpful assistent."
            content = [{
                "type": "image",
                "image": id2images[image_id],
                "max_pixels": self.max_image_pixels,
                "min_pixels": self.min_image_pixels
            } for image_id in image_ids]

        content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        num_visual_tokens = 0
        if image_inputs is not None:
            for image_input in image_inputs:
                num_visual_tokens += image_input.height * image_input.width \
                    // ((self.processor.image_processor.patch_size * self.processor.image_processor.merge_size) ** 2)
        
        if self.deploy_type == "api":
            image_sizes = [image.size for image in image_inputs]
            img_idx = 0
            for c in messages[-1]["content"]:
                if c["type"] == "image":
                    image = c["image"].resize(image_sizes[img_idx], Image.LANCZOS)
                    buffered = io.BytesIO()
                    image_format = "PNG" if image.format is None else image.format
                    image.save(buffered, format=image_format)
                    img_bytes = buffered.getvalue()
                    image_base64 = base64.b64encode(img_bytes).decode("utf-8")
                    base64_qwen = f"data:image;base64,{image_base64}"
                    c["type"] = "image_url"
                    c["image_url"] = {"url": base64_qwen}
                    # c["max_pixels"] = int(c["max_pixels"] ** 0.5)
                    # c["min_pixels"] = int(c["min_pixels"] ** 0.5)
                    del c["image"]
                    del c["max_pixels"]
                    del c["min_pixels"]
                    img_idx += 1
            return text_id, self.id2text[text_id], messages, None, None, num_visual_tokens
        else:
            return text_id, self.id2text[text_id], prompt, image_inputs, video_inputs, num_visual_tokens


@dataclass
class DataCollator(object):
    deploy_type: str
    processor: Optional[Callable] = None
    
    def __call__(self, batch):
        text_ids = []
        texts = []
        prompts = []
        image_inputs = []
        video_inputs = []
        num_visual_tokens = []
        for e in batch:
            text_ids.append(e[0])
            texts.append(e[1])
            prompts.append(e[2])
            image_inputs.append(e[3])
            video_inputs.append(e[4])
            num_visual_tokens.append(e[5])

        if self.deploy_type == "transformers":
            images = []
            videos = []
            for image_input, video_input in zip(image_inputs, video_inputs):
                if image_input is not None:
                    images.extend(image_input)
                if video_input is not None:
                    videos.extend(video_input)
            image_inputs = None if len(images) == 0 else images
            video_inputs = None if len(videos) == 0 else videos
            
            llm_inputs = self.processor(
                text=prompts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            )
        elif self.deploy_type == "vllm_local":
            llm_inputs = []
            for prompt, image_input, video_input in zip(prompts, image_inputs, video_inputs):
                mm_data = {}
                if image_input is not None:
                    mm_data["image"] = image_input
                if video_input is not None:
                    mm_data["video"] = video_input
                llm_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": mm_data,
                })
        elif self.deploy_type == "api":
            llm_inputs = prompts
        return text_ids, texts, llm_inputs, num_visual_tokens

