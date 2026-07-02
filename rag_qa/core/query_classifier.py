import json
import os

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

from base.config import single_config as config
from base.logger import single_logger as logger


class QueryClassifier:
    def __init__(self, model_path: str | None = None, pretrained_model_path: str | None = None):
        self.pre_trained_model_path = pretrained_model_path or config.BERT_PRETRAINED_MODEL
        self.model_path = model_path or config.QUERY_CLASSIFIER_MODEL
        self.device = self._select_device(config.MODEL_DEVICE)
        self.label_map = {"通用知识": 0, "专业咨询": 1}
        self.model = None

        tokenizer_source = self.model_path if self.model_path and os.path.exists(self.model_path) else self.pre_trained_model_path
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_source)
        self.load_model()

    def _select_device(self, configured_device: str) -> torch.device:
        configured_device = (configured_device or "auto").lower()
        if configured_device != "auto":
            if configured_device == "cuda" and not torch.cuda.is_available():
                logger.warning("MODEL_DEVICE=cuda 不可用，回退到 CPU")
                return torch.device("cpu")
            if configured_device == "mps" and not torch.backends.mps.is_available():
                logger.warning("MODEL_DEVICE=mps 不可用，回退到 CPU")
                return torch.device("cpu")
            return torch.device(configured_device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load_model(self):
        if self.model_path and os.path.exists(self.model_path):
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            logger.info(f"查询分类模型加载成功：{self.model_path}")
        else:
            self.model = BertForSequenceClassification.from_pretrained(self.pre_trained_model_path, num_labels=2)
            logger.warning(
                "未找到微调后的查询分类模型，已使用预训练 BERT 初始化二分类模型。"
                "请通过 QUERY_CLASSIFIER_MODEL 指向训练好的模型目录以获得稳定分类效果。"
            )
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"查询分类模型使用设备：{self.device}")

    def save_model(self):
        if not self.model_path:
            raise RuntimeError("QUERY_CLASSIFIER_MODEL 未配置，无法保存模型")
        os.makedirs(self.model_path, exist_ok=True)
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        logger.info(f"保存模型成功：{self.model_path}")

    def preprocess_data(self, texts, labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        return encodings, [self.label_map[label] for label in labels]

    def create_dataset(self, encodings, labels):
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        return Dataset(encodings, labels)

    def train_model(self, data_file: str | None = None):
        data_file = data_file or config.CLASSIFIER_TRAIN_DATA
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"数据集文件不存在：{data_file}")

        with open(data_file, "r", encoding="utf-8") as f:
            data = [json.loads(value) for value in f if value.strip()]

        texts = [item["query"] for item in data]
        labels = [item["label"] for item in data]

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        train_encodings, train_label_ids = self.preprocess_data(train_texts, train_labels)
        val_encodings, val_label_ids = self.preprocess_data(val_texts, val_labels)

        training_args = TrainingArguments(
            output_dir="./bert_results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./bert_logs",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            fp16=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.create_dataset(train_encodings, train_label_ids),
            eval_dataset=self.create_dataset(val_encodings, val_label_ids),
            compute_metrics=self.compute_metrics,
        )

        logger.info("开始训练 BERT 查询分类模型")
        trainer.train()
        self.save_model()
        self.evaluate_model(val_texts, val_label_ids)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": (predictions == labels).mean()}

    def evaluate_model(self, texts, labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        dataset = self.create_dataset(encodings, labels)
        predictions = Trainer(model=self.model).predict(dataset)
        pred_labels = np.argmax(predictions.predictions, axis=-1)

        logger.info("分类报告:")
        logger.info(classification_report(labels, pred_labels, target_names=["通用知识", "专业咨询"]))
        logger.info("混淆矩阵:")
        logger.info(confusion_matrix(labels, pred_labels))

    def predict_category(self, query):
        if self.model is None:
            logger.error("查询分类模型未加载，默认按通用知识处理")
            return "通用知识"

        encoding = self.tokenizer(query, truncation=True, padding=True, max_length=128, return_tensors="pt")
        encoding = {key: value.to(self.device) for key, value in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        return "专业咨询" if prediction == 1 else "通用知识"


if __name__ == "__main__":
    QueryClassifier().train_model()
