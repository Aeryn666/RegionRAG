import os
import argparse
import pandas as pd
from tqdm import tqdm
from colpali_engine.evaluation import RetrievalEvaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--topks", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--force_inference", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--infer_bbox", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--bbox_score_path", type=str, default=None)
    parser.add_argument("--bbox_score_method", type=str, default="max")
    parser.add_argument("--bbox_threshold", type=float, default=.0)
    parser.add_argument("--bbox_neighbor_range", type=int, default=1)
    parser.add_argument("--eval_save_root", type=str)
    parser.add_argument("--retrieval_result_name", type=str)
    parser.add_argument("--eval_iou_threshold", type=float, default=0.5)
    
    

    return parser.parse_args()


def retrieval():
    args = parse_args()
    model_path = args.model_path
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    topks = [int(i) for i in args.topks.split(',')]
    batch_size = args.batch_size
    force_inference = args.force_inference
    infer_bbox = args.infer_bbox
    bbox_score_path = args.bbox_score_path
    bbox_score_method = args.bbox_score_method
    bbox_threshold = args.bbox_threshold
    bbox_neighbor_range = list(range(-args.bbox_neighbor_range, args.bbox_neighbor_range + 1))
    eval_save_root = args.eval_save_root
    retrieval_results_path = os.path.join(eval_save_root, args.retrieval_result_name)
    image_inference_path = os.path.join(eval_save_root, "image.pth")
    text_inference_path = os.path.join(eval_save_root, "text.pth")
    bbox_num_process = 30
    eval_iou_threshold = args.eval_iou_threshold
    # bbox_num_process = 48
    batch_size = 16
    evaluator = RetrievalEvaluator(
        model_path, dataset_name, dataset_path, topks, batch_size,
        image_inference_path, text_inference_path,
        retrieval_results_path, force_inference,
        infer_bbox, bbox_score_path, bbox_score_method,
        bbox_threshold, bbox_neighbor_range, bbox_num_process,
        eval_iou_threshold
    )
    evaluator.run()


def search_retrieval():
    args = parse_args()
    model_path = args.model_path
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    topks = [int(i) for i in args.topks.split(',')]
    batch_size = args.batch_size
    force_inference = args.force_inference
    # infer_bbox = args.infer_bbox
    bbox_score_path = args.bbox_score_path
    bbox_score_method = args.bbox_score_method
    bbox_thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    bbox_neighbor_ranges = [list(range(-1,2)), list(range(-2,3))]
    eval_save_root = args.eval_save_root

    # retrieval_results_path = os.path.join(eval_save_root, args.retrieval_result_name)
    image_inference_path = os.path.join(eval_save_root, "image.pth")
    text_inference_path = os.path.join(eval_save_root, "text.pth")
    bbox_num_process = 40
    batch_size = 32

    all_results = []

    infer_bbox = False
    bbox_threshold = None
    bbox_neighbor_range = None
    retrieval_result_name="retrieval_results.json"
    retrieval_results_path = os.path.join(eval_save_root, retrieval_result_name)
    evaluator = RetrievalEvaluator(
        model_path, dataset_name, dataset_path, topks, batch_size,
        image_inference_path, text_inference_path,
        retrieval_results_path, force_inference,
        infer_bbox, bbox_score_path, bbox_score_method,
        bbox_threshold, bbox_neighbor_range, bbox_num_process,
        eval_iou_threshold=None
    )
    result = dict(
        retrieval_result_name=retrieval_result_name,
        bbox_threshold=bbox_threshold,
        bbox_neighbor_range=bbox_neighbor_range,
    )
    result.update(evaluator.run())
    all_results.append(result)

    infer_bbox = True
    for bbox_threshold in bbox_thresholds:
        for bbox_neighbor_range in bbox_neighbor_ranges:
            retrieval_result_name = f"box_{bbox_neighbor_range[-1]}_{bbox_score_method}_{bbox_threshold}_retrieval_results.json"
            retrieval_results_path = os.path.join(eval_save_root, retrieval_result_name)
            print(retrieval_results_path)
            evaluator = RetrievalEvaluator(
                model_path, dataset_name, dataset_path, topks, batch_size,
                image_inference_path, text_inference_path,
                retrieval_results_path, force_inference,
                infer_bbox, bbox_score_path, bbox_score_method,
                bbox_threshold, bbox_neighbor_range, bbox_num_process,
                eval_iou_threshold=None
            )
            result = dict(
                retrieval_result_name=retrieval_result_name,
                bbox_threshold=bbox_threshold,
                bbox_neighbor_range=bbox_neighbor_range[-1],
            )
            result.update(evaluator.run())
            all_results.append(result)
            print()

    df = pd.DataFrame(all_results)
    print(df)
    df.to_excel(os.path.join(eval_save_root, "search_retrieval.xlsx"), index=False)


def retrieval_datasets():
    args = parse_args()
    model_path = args.model_path
    model_name = model_path.split("/")[-1]
    # dataset_name = args.dataset_name
    # dataset_path = args.dataset_path
    dataset_names = [
        'mpdocvqa', 'arxivqa', 'chartqa', 'infovqa', 'plotqa', 'slidevqa',
        'vidore_arxivqa', 'vidore_docvqa', 'vidore_infovqa', 'vidore_shift', 'vidore_ai', 'vidore_energy', 'vidore_government', 'vidore_health', 'vidore_tabfquad', 'vidore_tatdqa'
    ]
    dataset_paths = [
        "data_dir/VisRAG/VisRAG-Ret-Test-MP-DocVQA",
        "data_dir/VisRAG/VisRAG-Ret-Test-ArxivQA",
        "data_dir/VisRAG/VisRAG-Ret-Test-ChartQA",
        "data_dir/VisRAG/VisRAG-Ret-Test-InfoVQA",
        "data_dir/VisRAG/VisRAG-Ret-Test-PlotQA",
        "data_dir/VisRAG/VisRAG-Ret-Test-SlideVQA",
        "data_dir/ViDoRe/arxivqa_test_subsampled",
        "data_dir/ViDoRe/docvqa_test_subsampled",
        "data_dir/ViDoRe/infovqa_test_subsampled",
        "data_dir/ViDoRe/shiftproject_test",
        "data_dir/ViDoRe/syntheticDocQA_artificial_intelligence_test",
        "data_dir/ViDoRe/syntheticDocQA_energy_test",
        "data_dir/ViDoRe/syntheticDocQA_government_reports_test",
        "data_dir/ViDoRe/syntheticDocQA_healthcare_industry_test",
        "data_dir/ViDoRe/tabfquad_test_subsampled",
        "data_dir/ViDoRe/tatdqa_test",
    ]
    topks = [int(i) for i in args.topks.split(',')]
    batch_size = args.batch_size
    force_inference = args.force_inference
    infer_bbox = args.infer_bbox
    bbox_score_path = args.bbox_score_path
    bbox_score_method = args.bbox_score_method
    bbox_threshold = args.bbox_threshold
    bbox_neighbor_range = list(range(-args.bbox_neighbor_range, args.bbox_neighbor_range + 1))

    bbox_num_process = 30
    eval_iou_threshold = args.eval_iou_threshold
    # bbox_num_process = 48
    batch_size = 64

    all_results = []
    for dataset_name, dataset_path in zip(dataset_names, dataset_paths):
        eval_save_root = f"work_dirs/eval_output/{model_name}/{dataset_name}"
        retrieval_results_path = os.path.join(eval_save_root, args.retrieval_result_name)
        image_inference_path = os.path.join(eval_save_root, "image.pth")
        text_inference_path = os.path.join(eval_save_root, "text.pth")
        evaluator = RetrievalEvaluator(
            model_path, dataset_name, dataset_path, topks, batch_size,
            image_inference_path, text_inference_path,
            retrieval_results_path, force_inference,
            infer_bbox, bbox_score_path, bbox_score_method,
            bbox_threshold, bbox_neighbor_range, bbox_num_process,
            eval_iou_threshold
        )
        result = dict(
            dataset_name=dataset_name,
        )
        result.update(evaluator.run())
        all_results.append(result)

    df = pd.DataFrame(all_results)
    print(df)
    df.to_excel(os.path.join(eval_save_root, "image_dataset_search_retrieval.xlsx"), index=False)

if __name__ == "__main__":
    retrieval_datasets()