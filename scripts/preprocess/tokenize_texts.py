import os

import MeCab

original_text_paths: list[str] = [
    "assets/jwtd_v2.0/gold_normalized_orig.txt",
    "assets/jwtd_v2.0/gold_normalized_corr.txt",
    "assets/jwtd_v2.0/gold_normalized_predicted_elyza_jp_8b.txt",
]

tokenized_text_paths = [
    path.replace(".txt", "_tokenized.txt") for path in original_text_paths
]

for original_text_path, tokenized_text_path in zip(
    original_text_paths, tokenized_text_paths
):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(tokenized_text_path), exist_ok=True)

    # Initialize MeCab tokenizer
    tokenizer = MeCab.Tagger("-Owakati")

    # Tokenize and save to file
    with (
        open(original_text_path, "r", encoding="utf-8") as in_file,
        open(tokenized_text_path, "w", encoding="utf-8") as out_file,
    ):
        for line in in_file:
            tokenized_line = tokenizer.parse(line.strip())
            out_file.write(tokenized_line)
    print(f"Tokenized text saved to {tokenized_text_path}")
