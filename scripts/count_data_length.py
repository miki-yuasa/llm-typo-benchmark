import os

text_data_path: str = "assets/jwtd_v2.0/gold_normalized_orig.txt"


# Count the length of strings in the text file
# Ignore \n
def count_data_length(file_path: str) -> int:
    """Count the length of strings in the text file"""
    total_length = 0
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            total_length += len(line.strip())
    return total_length


if __name__ == "__main__":
    # Get the length of the text data
    total_length = count_data_length(text_data_path)
    # Use scientific notation for large numbers
    total_length_str = "{:.2e}".format(total_length)
    print(f"Total length of text data: {total_length_str} characters")
