import os
import math
import torch
import argparse
import datetime
import pandas as pd
import torch.distributed as dist
from tqdm import tqdm
from colpali_engine.evaluation import GenerationEvaluator
from colpali_engine.utils.torch_utils import (
    get_world_size, get_rank,
    is_main_process, synchronize
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--deploy_type", type=str, default="transformers")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--eval_save_root", type=str, help='root to save both retrieval and generation results')
    parser.add_argument("--retrieval_result_name", type=str, help='name of retrieval result file')
    parser.add_argument("--response_name", type=str, help='name of generation response result file')
    parser.add_argument("--force_generate", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--topk_image", type=str)
    parser.add_argument("--max_image_pixels", type=int, default=1024*1024)
    parser.add_argument("--min_image_pixels", type=int, default=256*256)
    parser.add_argument("--max_bbox_pixels", type=int, default=1024*1024)
    parser.add_argument("--min_bbox_pixels", type=int, default=256*256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--infer_visual_type", type=str, default='image', choices=['image', 'bbox', 'image,bbox', 'bbox,image', 'oracle'])
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--api_num_processes", type=int, default=1)

    return parser.parse_args()


def generate():
    # dist.init_process_group(backend="gloo", timeout=datetime.timedelta(minutes=120))
    # local_rank = int(os.environ["LOCAL_RANK"])
    # torch.cuda.set_device(local_rank)

    args = parse_args()
    model_path = args.model_path
    model_name = args.model_path.split('/')[-1]
    deploy_type = args.deploy_type
    max_new_tokens = args.max_new_tokens
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    eval_save_root = args.eval_save_root
    retrieval_results_path = os.path.join(eval_save_root, args.retrieval_result_name)
    # image_inference_path = os.path.join(eval_save_root, "image.pth")
    # text_inference_path = os.path.join(eval_save_root, "text.pth")
    force_generate = args.force_generate
    topk_image = int(args.topk_image)
    max_image_pixels = args.max_image_pixels
    min_image_pixels = args.min_image_pixels
    max_bbox_pixels = args.max_bbox_pixels
    min_bbox_pixels = args.min_bbox_pixels
    batch_size = args.batch_size
    # response_name = args.retrieval_result_name.replace("retrieval_results", model_name).replace(".json", ".jsonl")
    response_path = os.path.join(eval_save_root, args.response_name)
    infer_visual_type = args.infer_visual_type
    vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
    api_num_processes = args.api_num_processes
    world_size = get_world_size()
    rank = get_rank()

    evaluator = GenerationEvaluator(
        model_path, deploy_type, max_new_tokens,
        dataset_name, dataset_path, retrieval_results_path,
        force_generate, topk_image, max_image_pixels, min_image_pixels,
        max_bbox_pixels, min_bbox_pixels, batch_size, response_path, infer_visual_type,
        vllm_tensor_parallel_size, api_num_processes, world_size, rank
    )
    evaluator.run()



def search_generate():
    dist.init_process_group(backend="gloo", timeout=datetime.timedelta(minutes=120))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    args = parse_args()
    model_path = args.model_path
    model_name = args.model_path.split('/')[-1]
    deploy_type = args.deploy_type
    max_new_tokens = args.max_new_tokens
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    eval_save_root = args.eval_save_root
    retrieval_results = pd.read_excel(os.path.join(eval_save_root, "search_retrieval.xlsx"))
    # retrieval_result_names = retrieval_results['retrieval_result_name'].to_list()
    # retrieval_results_path = os.path.join(eval_save_root, args.retrieval_result_name)
    # image_inference_path = os.path.join(eval_save_root, "image.pth")
    # text_inference_path = os.path.join(eval_save_root, "text.pth")
    force_generate = args.force_generate
    topk_images = [1]
    # topk_images = [int(topk) for topk in args.topk_image.split(',')]

    # max_image_pixels = args.max_image_pixels
    # min_image_pixels = args.min_image_pixels
    # max_bbox_pixels = args.max_bbox_pixels
    # min_bbox_pixels = args.min_bbox_pixels
    # pixels = [256*256, 448*448, 512*512, 768*768, 1024*1024]
    pixels = [256*256, 512*512]

    batch_size = args.batch_size
    # response_name = args.retrieval_result_name.replace("retrieval_results", model_name).replace(".json", ".jsonl")
    # response_path = os.path.join(eval_save_root, args.response_name)
    # infer_visual_type = args.infer_visual_type
    vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
    api_num_processes = args.api_num_processes
    world_size = get_world_size()
    rank = get_rank()

    all_results = []
    for index, retrieval_result in tqdm(retrieval_results.iterrows(), total=len(retrieval_results), desc="retrieval json"):
        retrieval_result_name = retrieval_result['retrieval_result_name']
        bbox_neighbor_range = retrieval_result['bbox_neighbor_range']
        bbox_threshold = retrieval_result['bbox_threshold']
        # if retrieval_result_name == "retrieval_results.json":
        #     infer_visual_types = ["image"]
        # else:
        #     # infer_visual_types = ["bbox", "image,bbox"]
        #     infer_visual_types = ["bbox"]
        #     # infer_visual_types = []

        infer_visual_types = ["oracle"]
        
        retrieval_results_path = os.path.join(eval_save_root, retrieval_result_name)
        for pixel in pixels:
            for topk_image in topk_images:
                for infer_visual_type in infer_visual_types:
                    if pixel != 256 * 256 and infer_visual_type == "image,bbox":
                        max_image_pixels_list = [256*256, pixel]
                        min_image_pixels_list = [256*256, pixel]
                        # max_image_pixels_list = [pixel]
                        # min_image_pixels_list = [pixel]
                    else:
                        max_image_pixels_list = [pixel]
                        min_image_pixels_list = [64*64]
                    max_bbox_pixels = pixel
                    min_bbox_pixels = 64*64
                    for max_image_pixels, min_image_pixels in zip(max_image_pixels_list, min_image_pixels_list):
                        # if topk_image == 8 and pixel == 1024*1024:
                        #     continue
                        if max_image_pixels == max_bbox_pixels:
                            response_name = f"{deploy_type}_{infer_visual_type}_{bbox_neighbor_range}_max_{bbox_threshold}_{model_name}_top{topk_image}_pixel{int(math.sqrt(pixel))}_min64.jsonl"
                        else:
                            response_name = f"{deploy_type}_{infer_visual_type}_{bbox_neighbor_range}_max_{bbox_threshold}_{model_name}_top{topk_image}_pixel256,{int(math.sqrt(pixel))}_min64.jsonl"
                        response_path = os.path.join(eval_save_root, response_name)
                        if is_main_process():
                            print(response_path)
                        evaluator = GenerationEvaluator(
                            model_path, deploy_type, max_new_tokens,
                            dataset_name, dataset_path, retrieval_results_path,
                            force_generate, topk_image, max_image_pixels, min_image_pixels,
                            max_bbox_pixels, min_bbox_pixels, batch_size, response_path, infer_visual_type,
                            vllm_tensor_parallel_size, api_num_processes, world_size, rank
                        )
                        generation_result = evaluator.run()
                        if is_main_process():
                            result = retrieval_result.to_dict()
                            result.update(dict(
                                pixel=int(math.sqrt(pixel)),
                                topk_image=topk_image,
                                infer_visual_type=infer_visual_type,
                            ))
                            result.update(generation_result)
                            all_results.append(result)
                        synchronize()
                        del evaluator
                        # torch.cuda.empty_cache()
                        if is_main_process():
                            print()
                        synchronize()

    if is_main_process():
        df = pd.DataFrame(all_results)
        print(df)
        df.to_excel(os.path.join(eval_save_root, f"{deploy_type}_dynamic_pixel_search_generation.xlsx"), index=False)
    synchronize()


if __name__ == "__main__":
    search_generate()
    # generate()
    # import os

    # folder_path = "../work_dirs/eval_output/data_vi_vc/mpdocvqa"
    # target_str = 'Qwen2.5-VL-7B'
    # prefix = 'transformers_'

    # for filename in os.listdir(folder_path):
    #     if target_str in filename:
    #         old_path = os.path.join(folder_path, filename)
    #         new_filename = prefix + filename
    #         new_path = os.path.join(folder_path, new_filename)
    #         os.rename(old_path, new_path)
    #         print(f'Renamed: {filename} --> {new_filename}')