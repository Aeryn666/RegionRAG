import re
from typing import List, Tuple, Optional
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform


def preprocess_text(text):
    # 替换换行符和制表符并去除前后空白
    text = text.replace('\n', ' ').replace('\t', ' ').strip()

    # 定义标点符号和模式
    punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
    period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
    comma_strip = re.compile(r"(\d)(\,)(\d)")

    # 定义缩略词映射
    contractions = {
        "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
        "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", 
        "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", 
        "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd", 
        "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", 
        "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", 
        "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", 
        "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", 
        "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", 
        "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", 
        "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", 
        "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd", 
        "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll", "someones": "someone's", 
        "somethingd": "something'd", "somethingd've": "something'd've", "something'dve": "something'd've", "somethingll": "something'll", 
        "thats": "that's", "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're", 
        "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll", 
        "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", 
        "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've", 
        "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've", 
        "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're", 
        "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", 
        "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", 
        "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", 
        "youll": "you'll", "youre": "you're", "youve": "you've"
    }

    # 定义数字映射和冠词
    manual_map = {
        'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    }
    articles = ['a', 'an', 'the']

    # 处理标点符号
    for p in punct:
        if (p + ' ' in text or ' ' + p in text) or (re.search(comma_strip, text) is not None):
            text = text.replace(p, '')
        else:
            text = text.replace(p, ' ')
    text = period_strip.sub("", text, re.UNICODE)

    # 处理数字和冠词
    words = text.lower().split()
    processed_words = []
    for word in words:
        word = manual_map.get(word, word)
        if word not in articles:
            processed_words.append(word)

    # 处理缩略词
    for idx, word in enumerate(processed_words):
        if word in contractions:
            processed_words[idx] = contractions[word]

    return ' '.join(processed_words)

def is_numeric_data(text):
    try:
        float(text)
        return True
    except:
        return False
    
def is_within_5_percent(responds, answer):
    # used for relaxed accuracy
    # 计算差距的百分比
    answer = float(answer)
    responds = float(responds)
    diff_percentage = abs((responds - answer) / answer) * 100
    
    # 判断是否不超过 5%
    return diff_percentage <= 5


def check_responses(dataset_name, responds, answer, query):
    #pre-process
    correct = 0
    if (dataset_name.lower() == 'chartqa'):
        responds = preprocess_text(responds)
        answer = preprocess_text(answer)
        if ('%' in responds and '%' not in answer):
            responds = responds.replace('%', '')
        if ('%' not in responds and '%' in answer):
            answer = answer.replace('%', '')
        # print(f"query: {query}")
        # print(f"responds:{responds}")
        # print(f"answer:{answer}")
        if (responds == answer):
            correct = 1
        elif(is_numeric_data(responds) and is_numeric_data(answer) and answer != '0' and is_within_5_percent(responds, answer)):
            correct = 1
    elif (dataset_name.lower() == 'arxivqa'):
        if responds and answer:
            responds = responds[0].upper()
            answer = answer[0].upper()
            # print(f"query: {query}")
            # print(f"responds:{responds}")
            # print(f"answer:{answer}")
            if (responds == answer):
                correct = 1
        else:
            correct = 0
    elif (dataset_name.lower() == 'plotqa'):
        responds = preprocess_text(responds)
        is_str = 1
        if (type(answer) != str):
            is_str = 0
            answer = str(answer)
        answer = preprocess_text(answer)
        if ('%' in responds and '%' not in answer):
            responds = responds.replace('%', '')
        if ('%' not in responds and '%' in answer):
            answer = answer.replace('%', '')
        # print(f"query: {query}")
        # print(f"responds:{responds}")
        # print(f"answer:{answer}")
        if (responds == answer):
            correct = 1
        elif(is_numeric_data(responds) and (not is_str) and float(answer) != 0.0 and is_within_5_percent(responds, answer)):
            correct = 1
    elif (dataset_name.lower() in ['mp-docvqa', 'mpdocvqa']):
        responds = preprocess_text(responds)
        if (not isinstance(answer, list)):
            answer = [answer]
        for i, answer_item in enumerate(answer):
            answer[i] = preprocess_text(answer_item)
        if ('%' in responds and '%' not in answer[0]):
            responds = responds.replace('%', '')
        if ('%' not in responds and '%' in answer[0]):
            answer = [answer_item.replace('%', '') for answer_item in answer]
        # print(f"query: {query}")
        # print(f"responds:{responds}")
        # print(f"answer:{answer}")
        for answer_item in answer:
            if (responds == answer_item):
                correct = 1
                break
    elif (dataset_name.lower() == 'slidevqa'):
        responds = preprocess_text(responds)
        answer = preprocess_text(answer)
        if ('%' in responds and '%' not in answer):
            responds = responds.replace('%', '')
        if ('%' not in responds and '%' in answer):
            answer = answer.replace('%', '')
        # print(f"query: {query}")
        # print(f"responds:{responds}")
        # print(f"answer:{answer}")
        if (responds == answer):
            correct = 1
    elif (dataset_name.lower() == 'infovqa'):
        responds = preprocess_text(responds)
        if (not isinstance(answer, list)):
            answer = [answer]
        for i, answer_item in enumerate(answer):
            answer[i] = preprocess_text(answer_item)
        if ('%' in responds and '%' not in answer[0]):
            responds = responds.replace('%', '')
        if ('%' not in responds and '%' in answer[0]):
            answer = [answer_item.replace('%', '') for answer_item in answer]
        # print(f"query: {query}")
        # print(f"responds:{responds}")
        # print(f"answer:{answer}")
        for answer_item in answer:
            if (responds == answer_item):
                correct = 1
                break
    
    return correct, responds, answer



class BBoxMerger:
    @staticmethod
    def merge(
        bboxes: List[List[float]], 
        method: str = "nms", 
        **kwargs
    ) -> List[List[float]]:
        """
        多策略bbox合并入口函数
        参数：
            bboxes: 待合并的bbox列表，每个bbox格式为[x1,y1,x2,y2]
            method: 合并策略，可选:
                - 'nms'：非极大值抑制
                - 'density'：基于密度的空间聚类
                - 'hierarchical'：层次聚类
                - 'grid'：基于网格的合并
                - 'iou'：交并比合并
            **kwargs: 各策略的定制参数

        返回：
            合并后的bbox列表
        """
        if len(bboxes) == 0:
            return []

        method_func = {
            "nms": BBoxMerger._nms_merge,
            "density": BBoxMerger._density_based_merge,
            "hierarchical": BBoxMerger._hierarchical_merge,
            "grid": BBoxMerger._grid_based_merge,
            "iou": BBoxMerger._iou_merge,
        }.get(method.lower(), BBoxMerger._nms_merge)

        return method_func(np.array(bboxes), **kwargs)

    @staticmethod
    def _nms_merge(
        bboxes: np.ndarray, 
        iou_thresh: float = 0.5,
        score: Optional[np.ndarray] = None
    ) -> List[List[float]]:
        """
        改进版非极大值抑制(NMS)
        参数：
            bboxes: [N,4]矩阵
            iou_thresh: IoU阈值
            score: 各bbox的置信分数，若为None则按面积排序
        """
        if score is None:
            score = (bboxes[:,2]-bboxes[:,0]) * (bboxes[:,3]-bboxes[:,1])
        
        sorted_indices = np.argsort(score)[::-1]
        keep = []
        
        while sorted_indices.size > 0:
            current = sorted_indices[0]
            keep.append(current)
            
            current_box = bboxes[current]
            other_boxes = bboxes[sorted_indices[1:]]
            
            # 计算IoU
            xx1 = np.maximum(current_box[0], other_boxes[:,0])
            yy1 = np.maximum(current_box[1], other_boxes[:,1])
            xx2 = np.minimum(current_box[2], other_boxes[:,2])
            yy2 = np.minimum(current_box[3], other_boxes[:,3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            area_current = (current_box[2]-current_box[0])*(current_box[3]-current_box[1])
            area_others = (other_boxes[:,2]-other_boxes[:,0])*(other_boxes[:,3]-other_boxes[:,1])
            iou = intersection / (area_current + area_others - intersection)
            
            # 保留低重叠框
            retain_indices = np.where(iou <= iou_thresh)[0]
            sorted_indices = sorted_indices[retain_indices + 1]
            
        return bboxes[keep].tolist()

    @staticmethod
    def _density_based_merge(
        bboxes: np.ndarray,
        eps: float = 0.2,  # 基于归一化距离的参数
        min_samples: int = 2
    ) -> List[List[float]]:
        """
        基于密度的空间聚类（DBSCAN）
        参数：
            eps: 邻域半径（按图像尺寸归一化）
            min_samples: 核心点所需最小样本数
        """
        # 计算归一化中心坐标
        centers = np.stack([
            (bboxes[:,0] + bboxes[:,2])/2,
            (bboxes[:,1] + bboxes[:,3])/2
        ], axis=1)
        
        # 动态计算eps的像素值
        img_width = bboxes[:,2].max()
        img_height = bboxes[:,3].max()
        eps_pixel = eps * np.sqrt(img_width**2 + img_height**2)
        
        # 执行DBSCAN聚类
        clustering = DBSCAN(eps=eps_pixel, min_samples=min_samples).fit(centers)
        
        merged = []
        for label in np.unique(clustering.labels_):
            if label == -1:  # 噪声点单独保留
                continue
            cluster = bboxes[clustering.labels_ == label]
            merged.append([
                cluster[:,0].min(), cluster[:,1].min(),
                cluster[:,2].max(), cluster[:,3].max()
            ])
        
        # 保留未聚类的点
        noise = bboxes[clustering.labels_ == -1]
        if len(noise) > 0:
            merged.extend(noise.tolist())
            
        return merged

    @staticmethod
    def _hierarchical_merge(
        bboxes: np.ndarray,
        threshold: float = 0.3
    ) -> List[List[float]]:
        """
        层次聚类合并
        参数：
            threshold: 合并距离阈值（IoU）
        """
        if len(bboxes) == 1:
            return bboxes.tolist()
        
        # 计算IoU距离矩阵
        iou_matrix = 1 - BBoxMerger._pairwise_iou(bboxes)
        condensed_dist = squareform(iou_matrix)
        
        # 执行层次聚类
        from scipy.cluster.hierarchy import fcluster, linkage
        Z = linkage(condensed_dist, method='average')
        clusters = fcluster(Z, t=threshold, criterion='distance')
        
        merged = []
        for label in np.unique(clusters):
            cluster = bboxes[clusters == label]
            merged.append([
                cluster[:,0].min(), cluster[:,1].min(),
                cluster[:,2].max(), cluster[:,3].max()
            ])
        return merged

    @staticmethod
    def _grid_based_merge(
        bboxes: np.ndarray,
        grid_size: Tuple[int, int] = (3,3)
    ) -> List[List[float]]:
        """
        基于图像网格的合并策略
        参数：
            grid_size: 将图像划分为 (rows, cols) 的网格
        """
        img_width = bboxes[:,2].max()
        img_height = bboxes[:,3].max()
        
        grid_w = img_width / grid_size[1]
        grid_h = img_height / grid_size[0]
        
        grid_bboxes = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                x1 = j * grid_w
                y1 = i * grid_h
                x2 = (j+1) * grid_w
                y2 = (i+1) * grid_h
                grid_bboxes.append([x1, y1, x2, y2])
                
        return grid_bboxes  # 可根据实际需求添加与候选框的交集判断

    @staticmethod
    def _iou_merge(
        bboxes: np.ndarray,
        iou_thresh: float = 0.7
    ) -> List[List[float]]:
        """
        渐进式IoU合并
        参数：
            iou_thresh: 合并的IoU阈值
        """
        merged = []
        for box in bboxes:
            matched = False
            for i, m in enumerate(merged):
                # 计算IoU
                inter_x1 = max(m[0], box[0])
                inter_y1 = max(m[1], box[1])
                inter_x2 = min(m[2], box[2])
                inter_y2 = min(m[3], box[3])
                
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    area_m = (m[2]-m[0])*(m[3]-m[1])
                    area_b = (box[2]-box[0])*(box[3]-box[1])
                    iou = inter_area / (area_m + area_b - inter_area)
                    
                    if iou > iou_thresh:
                        # 合并两个框
                        merged[i] = [
                            min(m[0], box[0]),
                            min(m[1], box[1]),
                            max(m[2], box[2]),
                            max(m[3], box[3])
                        ]
                        matched = True
                        break
            if not matched:
                merged.append(box.tolist())
        return merged

    @staticmethod
    def _pairwise_iou(bboxes: np.ndarray) -> np.ndarray:
        """
        计算bboxes之间的IoU矩阵
        """
        area = (bboxes[:,2] - bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1])
        
        xx1 = np.maximum(bboxes[:, None, 0], bboxes[:, 0])
        yy1 = np.maximum(bboxes[:, None, 1], bboxes[:, 1])
        xx2 = np.minimum(bboxes[:, None, 2], bboxes[:, 2])
        yy2 = np.minimum(bboxes[:, None, 3], bboxes[:, 3])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = area[:, None] + area - inter
        
        return inter / (union + 1e-8)
