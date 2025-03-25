import errant
from errant import Annotator
import spacy
import ollama
import json
import asyncio
from collections import defaultdict


def calculate_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Calculate precision, recall, and F0.5 scores."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f05 = (
        ((1 + 0.5**2) * precision * recall) / ((0.5**2 * precision) + recall)
        if (precision + recall) > 0
        else 0
    )
    return precision, recall, f05


async def evaluate_corrections(
    orig_text_path: str,
    corr_text_path: str,
    pred_text_path: str,
    spacy_model_name: str,
    stats_save_path: str,
) -> dict[str, float]:

    # Load text files
    with open(orig_text_path, "r", encoding="utf-8") as f:
        orig_texts = f.readlines()
    with open(corr_text_path, "r", encoding="utf-8") as f:
        corr_texts = f.readlines()
    with open(pred_text_path, "r", encoding="utf-8") as f:
        pred_texts = f.readlines()

    results = defaultdict(int)

    nlp = spacy.load(spacy_model_name)
    annotator: Annotator = errant.load("en", nlp)

    for orig_text, corr_text, pred_text in zip(orig_texts, corr_texts, pred_texts):

        # Create ERRANT annotations
        source = annotator.parse(orig_text, tokenise=True)
        gold = annotator.parse(corr_text, tokenise=True)
        predicted = annotator.parse(pred_text, tokenise=True)

        # Get edits
        gold_edits = annotator.annotate(source, gold)
        pred_edits = annotator.annotate(source, predicted)

        # Compare edits
        for gold_edit in gold_edits:
            if any(pred_edit == gold_edit for pred_edit in pred_edits):
                results["tp"] += 1
            else:
                results["fn"] += 1

        for pred_edit in pred_edits:
            if not any(gold_edit == pred_edit for gold_edit in gold_edits):
                results["fp"] += 1

    # Calculate metrics
    precision, recall, f05 = calculate_metrics(
        results["tp"], results["fp"], results["fn"]
    )

    # Save results
    stats = {
        "precision": precision,
        "recall": recall,
        "f0.5": f05,
        "true_positives": results["tp"],
        "false_positives": results["fp"],
        "false_negatives": results["fn"],
    }

    with open(stats_save_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    # Load the test set
    # Here is a sample entry
    # {"page": "326", "title": "アーミッシュ", "pre_rev": "19336024", "post_rev": "27253461", "pre_text": "そのため自動車は運転しないが。", "post_text": "そのため自動車は運転しない。", "diffs": [{"pre_str": "が", "post_str": "", "pre_bart_likelihood": -21.22, "post_bart_likelihood": -10.74, "category": "insertion_a"}], "lstm_average_likelihood": -2.27}
    orig_text_path: str = "assets/jwtd_v2.0/gold_normalized_orig.txt"
    corr_text_path: str = "assets/jwtd_v2.0/gold_normalized_corr.txt"
    pred_text_path: str = "assets/jwtd_v2.0/gold_normalized_predicted_elyza_jp_8b.txt"
    stats_save_path: str = "out/eval/elyze_jp_8b_stats.json"

    spacy_model_name: str = "ja_core_news_md"

    # Pull the model from Ollama

    stats = asyncio.run(
        evaluate_corrections(
            orig_text_path,
            corr_text_path,
            pred_text_path,
            spacy_model_name,
            stats_save_path,
        )
    )
    print(f"Precision: {stats['precision']:.4f}")
    print(f"Recall: {stats['recall']:.4f}")
    print(f"F0.5: {stats['f0.5']:.4f}")
