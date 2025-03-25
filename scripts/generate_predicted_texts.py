original_text_path: str = "assets/jwtd_v2.0/test_normalized_orig.txt"
predicted_text_path: str = "assets/jwtd_v2.0/test_normalized_predicted_elyza_jp_8b.txt"

ollama_model_name: str = "hf.co/elyza/Llama-3-ELYZA-JP-8B-GGUF:Q4_K_M"
ollama_server_url: str = "http://localhost:11434"

import requests
import json
from pathlib import Path
from typing import List


def get_ollama_prediction(text: str, model: str, url: str) -> str:
    """Get prediction from Ollama server"""
    prompt = "Correct any grammatical or typing errors in the following Japanese text. Only output the corrected text. No preface. No comments."

    payload = {"model": model, "prompt": f"{prompt}\n\n{text}", "stream": False}

    response = requests.post(f"{url}/api/generate", json=payload)
    if response.status_code != 200:
        raise Exception(f"Ollama request failed with status {response.status_code}")

    return response.json()["response"].strip()


def process_file(input_path: str, output_path: str, model: str, url: str) -> None:
    """Process input file and write predictions to output file"""
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(input_path, "r", encoding="utf-8") as in_file,
        open(output_path, "w", encoding="utf-8") as out_file,
    ):
        for line in in_file:
            text = line.strip()
            if text:  # Skip empty lines
                prediction = get_ollama_prediction(text, model, url)
                out_file.write(f"{prediction}\n")
            else:
                out_file.write("\n")


def main():
    process_file(
        input_path=original_text_path,
        output_path=predicted_text_path,
        model=ollama_model_name,
        url=ollama_server_url,
    )


if __name__ == "__main__":
    main()
