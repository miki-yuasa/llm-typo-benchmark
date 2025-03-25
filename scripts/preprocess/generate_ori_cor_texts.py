import json

if __name__ == "__main__":
    # Load the JSONL file
    # Here is a sample entry
    # {"page": "326", "title": "アーミッシュ", "pre_rev": "19336024", "post_rev": "27253461", "pre_text": "そのため自動車は運転しないが。", "post_text": "そのため自動車は運転しない。", "diffs": [{"pre_str": "が", "post_str": "", "pre_bart_likelihood": -21.22, "post_bart_likelihood": -10.74, "category": "insertion_a"}], "lstm_average_likelihood": -2.27}
    input_path: str = "assets/jwtd_v2.0/gold_normalized.jsonl"
    output_orig_path: str = input_path.replace(".jsonl", "_orig.txt")
    output_corr_path: str = input_path.replace(".jsonl", "_corr.txt")

    # Open both output files
    with (
        open(output_orig_path, "w", encoding="utf-8") as orig_file,
        open(output_corr_path, "w", encoding="utf-8") as corr_file,
    ):

        # Read the input JSONL file
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # Write pre_text and post_text to respective files
                orig_file.write(data["pre_text"] + "\n")
                corr_file.write(data["post_text"] + "\n")
