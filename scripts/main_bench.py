import errant
from errant import Annotator
import spacy
import ollama
from ollama import AsyncClient
import json
import asyncio
from collections import defaultdict


async def get_model_correction(
    client: ollama.AsyncClient, text: str, ollama_model_name: str
) -> str:
    """Get correction from the LLM model."""
    response = await client.chat(
        model=ollama_model_name,
        messages=[
            {
                "role": "system",
                "content": "Correct any grammatical or typing errors in the following Japanese text. Only output the corrected text.",
            },
            {"role": "user", "content": text},
        ],
    )
    return response["message"]["content"].strip()


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
    ollama_model_name: str,
    ollama_server_url: str,
    test_text_jsonl_path: str,
    spacy_model_name: str,
    stats_save_path: str,
) -> dict[str, float]:
    # Initialize Ollama client
    client = ollama.AsyncClient(host=ollama_server_url)

    # Load test data
    with open(test_text_jsonl_path, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    results = defaultdict(int)

    nlp = spacy.load(spacy_model_name)
    annotator: Annotator = errant.load("ja", nlp)

    for entry in test_data:
        source_text = entry["pre_text"]
        gold_text = entry["post_text"]

        # Get model prediction
        predicted_text = await get_model_correction(
            client, source_text, ollama_model_name
        )

        # Create ERRANT annotations
        source = annotator.parse(source_text)
        gold = annotator.parse(gold_text)
        predicted = annotator.parse(predicted_text)

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
    test_text_jsonl_path: str = "assets/jwtd_v2.0/test_normalized.jsonl"
    stats_save_path: str = "out/eval/elyze_jp_8b_stats.json"

    # Define the models
    ollama_model_name: str = "hf.co/elyza/Llama-3-ELYZA-JP-8B-GGUF:Q4_K_M"
    ollama_server_url: str = "http://localhost:11434"
    spacy_model_name: str = "ja_core_news_md"

    # Pull the model from Ollama
    ollama.pull(ollama_model_name)

    stats = asyncio.run(
        evaluate_corrections(
            ollama_model_name,
            ollama_server_url,
            test_text_jsonl_path,
            spacy_model_name,
            stats_save_path,
        )
    )
    print(f"Precision: {stats['precision']:.4f}")
    print(f"Recall: {stats['recall']:.4f}")
    print(f"F0.5: {stats['f0.5']:.4f}")
