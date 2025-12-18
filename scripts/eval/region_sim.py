import os
import argparse
import pandas as pd
from tqdm import tqdm
from colpali_engine.evaluation import RegionSimEvaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_path", type=str)
    # parser.add_argument("--topks", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--force_inference", type=lambda x: x.lower() == "true", default=False)
    # parser.add_argument("--infer_bbox", type=lambda x: x.lower() == "true", default=False)
    # parser.add_argument("--bbox_score_path", type=str, default=None)
    # parser.add_argument("--bbox_score_method", type=str, default="max")
    # parser.add_argument("--bbox_threshold", type=float, default=.0)
    # parser.add_argument("--bbox_neighbor_range", type=int, default=1)
    parser.add_argument("--eval_save_root", type=str)
    # parser.add_argument("--retrieval_result_name", type=str)
    # parser.add_argument("--eval_iou_threshold", type=float, default=0.5)
    parser.add_argument("--is_visualization", type=lambda x: x.lower() == "true", default=False)
    
    

    return parser.parse_args()


def retrieval():
    args = parse_args()
    model_path = args.model_path
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    # topks = [int(i) for i in args.topks.split(',')]
    batch_size = args.batch_size
    force_inference = args.force_inference
    # infer_bbox = args.infer_bbox
    # bbox_score_path = args.bbox_score_path
    # bbox_score_method = args.bbox_score_method
    # bbox_threshold = args.bbox_threshold
    # bbox_neighbor_range = list(range(-args.bbox_neighbor_range, args.bbox_neighbor_range + 1))
    eval_save_root = args.eval_save_root
    os.makedirs(eval_save_root, exist_ok=True)
    # retrieval_results_path = os.path.join(eval_save_root, args.retrieval_result_name)
    image_inference_path = os.path.join(eval_save_root, "image.pth")
    text_inference_path = os.path.join(eval_save_root, "text.pth")
    # bbox_num_process = 30
    # eval_iou_threshold = args.eval_iou_threshold
    # bbox_num_process = 48
    batch_size = 4
    is_visualization = args.is_visualization
    evaluator = RegionSimEvaluator(
        model_path, dataset_name, dataset_path, batch_size,
        image_inference_path, text_inference_path,
        force_inference, is_visualization
        # infer_bbox, bbox_score_path, bbox_score_method,
        # bbox_threshold, bbox_neighbor_range, bbox_num_process,
        # eval_iou_threshold
    )
    evaluator.run()


if __name__ == "__main__":
    retrieval()