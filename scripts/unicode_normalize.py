import json
import os
import unicodedata

jsonl_save_path: str = "assets/jwtd_v2.0/train.jsonl"


def normalize_text(text: str) -> str:
    """Normalize Unicode text using NFKC normalization."""
    return unicodedata.normalize("NFKC", text)


def normalize_jsonl_file(input_path: str) -> None:
    """Read JSONL file, normalize text fields, and save to a new file."""
    # Create output path by inserting '_normalized' before the extension
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_normalized{ext}"

    normalized_data = []

    # Read and process the JSONL file
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())

            # Normalize all string values in the JSON object
            def normalize_dict(d):
                for key, value in d.items():
                    if isinstance(value, str):
                        d[key] = normalize_text(value)
                    elif isinstance(value, dict):
                        normalize_dict(value)
                    elif isinstance(value, list):
                        d[key] = [
                            normalize_text(item) if isinstance(item, str) else item
                            for item in value
                        ]
                    else:
                        pass
                return d

            normalized_data.append(normalize_dict(data))

    # Write normalized data to new file
    with open(output_path, "w", encoding="utf-8") as f:
        for item in normalized_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    normalize_jsonl_file(jsonl_save_path)
    print(
        f"Normalized file saved to: {os.path.splitext(jsonl_save_path)[0]}_normalized.jsonl"
    )
