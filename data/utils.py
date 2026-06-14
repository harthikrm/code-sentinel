import json

def load_examples(file_path, n):
    """..."""
    data = []
    counter = 0
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
            counter += 1
            if counter == n:
                break
    return data

def format_prompt(row):
    """..."""
    return f'''[INST] Review the following code change and identify issues:
Language: {row["lang"]}
Diff: {row["hunk"]}
Provide specific, actionable feedback.[/INST]'''