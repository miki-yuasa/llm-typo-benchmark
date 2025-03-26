# llm-typo-benchmark
Benchmarking for LLM typo checking

## Usage
1. Download the dataset under `/assets`.
2. Run `scripts/preprocess/unicode_normalize.py` to normalize the dataset.
3. Run `scripts/preprocess/generate_ori_cor_texts.py` to generate original and corrected texts.
4. Run `scripts/preprocess/tokenize_texts.py` to tokenize the texts.
5. `cd assets/jwed_v2.0`
6. Run `python3 -m spacy download en_core_web_sm` to download the spaCy model.
7. Run `errant_parallel -orig gold_normalized_orig_tokenized.txt -cor gold_normalized_predicted_tokenized.txt -out gold_pred.m2` to generate the reference M2 file.
8. Run `errant_parallel -orig gold_normalized_orig_tokenized.txt -cor gold_normalized_predicted_elyza_jp_8b_tokenized.txt -out gold_pred.m2` to generate the predicted M2 file.
9. Run `errant_compare -hyp gold_pred.m2 -ref gold_ref.m2 -v` to compare the predicted M2 file with the reference M2 file.

The procedures above can be run using the following command:
```bash
$ python3 scripts/preprocess/unicode_normalize.py
$ python3 scripts/preprocess/generate_ori_cor_texts.py
$ python3 scripts/preprocess/tokenize_texts.py
$ cd assets/jwed_v2.0
$ python3 -m spacy download en_core_web_sm
$ errant_parallel -orig gold_normalized_orig_tokenized.txt -cor gold_normalized_predicted_tokenized.txt -out gold_pred.m2
$ errant_parallel -orig gold_normalized_orig_tokenized.txt -cor gold_normalized_predicted_elyza_jp_8b_tokenized.txt -out gold_pred.m2
$ errant_compare -hyp gold_pred.m2 -ref gold_ref.m2 -v
```