import json
from preprocess import is_valid_comment, format_example

def process_file(input_file_path, output_file_path):
    """
    process_file(input_file_path, output_file_path):

    This function takes input and output file paths for data and preprocess the data using the funcitons we built in preprocess.py

    input: file paths
    output: JSONL line-by-line processed as a JSON key-value pair
    """
    passed = 0
    failed = 0

    with open(input_file_path, "r") as infile, open(output_file_path, "w") as outfile:
        for line in infile:
            row = json.loads(line)
            if not is_valid_comment(row["comment"]):
                failed += 1
                continue
            formatted = format_example(row)
            outfile.write(json.dumps({"text": formatted}) + "\n")
            passed += 1
    
    print(f"Number of passed rows: {passed}")
    print(f"Number of failed rows: {failed}")

if __name__ == "__main__":
    input_files = ["ref-train", "ref-test", "ref-valid"]
    output_files = ["processed-train", "processed-test", "processed-valid"]
    for input_file, output_file in zip(input_files, output_files):
        process_file(
            f"/Users/harthikmallichetty/Desktop/code-sentinel-data-source/{input_file}.jsonl",
            f"/Users/harthikmallichetty/Desktop/code-sentinel-data-source/{output_file}.jsonl"
        )