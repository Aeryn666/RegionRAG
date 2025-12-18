import os
import torch
import _codecs
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    PreTrainedModel,
    TrainingArguments,
)

from colpali_engine.collators import CorpusQueryCollator, VisualRetrieverCollator
from colpali_engine.loss.late_interaction_losses import (
    ColbertLoss,
)
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer, FlairTrainer
from colpali_engine.utils.gpu_stats import print_gpu_utilization, print_summary
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import check_ckpt_exists


@dataclass
class ColModelTrainingConfig:
    model: Union[PreTrainedModel, PeftModel]
    processor: BaseVisualRetrieverProcessor
    tr_args: Optional[TrainingArguments] = None
    output_dir: Optional[str] = None
    max_length: int = 256
    run_eval: bool = True
    run_train: bool = True
    peft_config: Optional[LoraConfig] = None
    loss_func: Optional[Callable] = ColbertLoss()
    dataset_loading_func: Optional[Callable] = None
    eval_dataset_loader: Optional[Dict[str, Callable]] = None
    pretrained_peft_model_name_or_path: Optional[str] = None
    trainer: Optional[Callable] = ContrastiveTrainer
    """
    Config class used for training a ColVision model.
    """

    def __post_init__(self):
        """
        Initialize the model and tokenizer if not provided
        """
        # if self.output_dir is None:
        #     sanitized_name = str(self.model.name_or_path).replace("/", "_")
        #     self.output_dir = f"./models/{sanitized_name}"

        if self.tr_args is None:
            print("No training arguments provided. Using default.")
            self.tr_args = TrainingArguments(output_dir=self.output_dir)
        # elif self.tr_args.output_dir is None or self.tr_args.output_dir == "trainer_output":
        #     self.tr_args.output_dir = self.output_dir
        #     self.tr_args.logging_dir = self.tr_args.logging_dir.replace("trainer_output", self.output_dir)

        if isinstance(self.tr_args.learning_rate, str):
            print("Casting learning rate to float")
            self.tr_args.learning_rate = float(self.tr_args.learning_rate)

        self.tr_args.remove_unused_columns = False

        if self.pretrained_peft_model_name_or_path is not None:
            print("Loading pretrained PEFT model")
            self.model.load_adapter(self.pretrained_peft_model_name_or_path, is_trainable=True)

        if self.peft_config is not None:
            print("Configurating PEFT model")
            if self.pretrained_peft_model_name_or_path is None:
                self.model = get_peft_model(self.model, self.peft_config)
                self.model.print_trainable_parameters()
            else:
                print(f"Adapter already loaded from {self.pretrained_peft_model_name_or_path}. Not overwriting.")

    print_gpu_utilization()


class ColModelTraining:
    """
    Class that contains the training and evaluation logic for a ColVision model.
    """

    def __init__(self, config: ColModelTrainingConfig) -> None:
        self.config = config
        self.model = self.config.model
        self.current_git_hash = os.popen("git rev-parse HEAD").read().strip()
        self.dataset = self.config.dataset_loading_func

        if isinstance(self.dataset, Tuple): # 普通BEIR格式包括Corpus、Queries、Qrels三个部分
            print("Dataset has BEIR/hard negatives format. Using CorpusQueryCollator.")
            corpus_format = self.dataset[2]
            neg_dataset = self.dataset[1]
            self.dataset = self.dataset[0]
            self.collator = CorpusQueryCollator(
                processor=self.config.processor,
                max_length=self.config.max_length,
                image_dataset=neg_dataset,
                mined_negatives=True,
                corpus_format=corpus_format,
            )
        else:
            print("Dataset has QA format. Using VisualRetrieverCollator.")
            self.collator = VisualRetrieverCollator(
                processor=self.config.processor,
                max_length=self.config.max_length,
            )
        
        self.trainer_cls = getattr(self.config, "trainer", ContrastiveTrainer)

    def train(self) -> None:
        if isinstance(self.collator, CorpusQueryCollator) and self.collator.mined_negatives:
            print("Training with hard negatives")
        else:
            print("Training with in-batch negatives")

        trainer = self.trainer_cls(
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            # eval_dataset=self.dataset["train"],
            args=self.config.tr_args,
            data_collator=self.collator,
            loss_func=self.config.loss_func,
            is_vision_model=self.config.processor is not None,
        )

        trainer.args.remove_unused_columns = False
        if check_ckpt_exists(self.config.output_dir): # 检查 output_dir 目录里是否存在至少一个有效的 checkpoint，决定是断点续训还是从头训练
            allowlist = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, _codecs.encode]
            allowlist += [type(np.dtype(np.uint32))]
            torch.serialization.add_safe_globals(allowlist)
            result = trainer.train(resume_from_checkpoint=self.config.tr_args.resume_from_checkpoint) # 启动训练，并从 checkpoint 继续
        else:
            result = trainer.train() # 没有可用ckpt，于是从头开始训练
        print_summary(result) # 打印训练结果摘要

    def eval(self) -> None:
        raise NotImplementedError("Evaluation is not implemented yet.")

    def save(self):
        """
        Save the model with its training config, as well as the tokenizer and processor if provided.
        """
        self.model.save_pretrained(self.config.output_dir)
        self.config.processor.save_pretrained(self.config.output_dir)

        # Save git hash of the commit at beginning of training
        with open(f"{self.config.output_dir}/git_hash.txt", "w") as f:
            f.write(self.current_git_hash)
