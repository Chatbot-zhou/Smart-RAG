import json
from pathlib import Path

from legal_brain.config import settings
from legal_brain.storage.mysql import LegalMySQLStore


def run_ragas_evaluation(dataset_path: Path, run_name: str = "legal-rag-eval") -> int:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise RuntimeError("RAGAS 评估生产依赖缺失：需要 ragas 和 datasets。") from exc

    records = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset = Dataset.from_list(records)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    metrics = result.to_pandas().mean(numeric_only=True).to_dict()
    store = LegalMySQLStore()
    run_id = store.record_ragas_eval_run(
        {
            "run_name": run_name,
            "dataset_path": str(dataset_path),
            "metrics": metrics,
            "corpus_version": settings.corpus_version,
        }
    )
    store.close()
    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for 智慧法务大脑.")
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--run-name", default="legal-rag-eval")
    args = parser.parse_args()
    print(run_ragas_evaluation(args.dataset_path, args.run_name))
