# Compare baseline, fine-tuned, and merged checkpoints across evaluation metrics.
import json
import os
from bert_score import score
from mlx_lm import load, generate
from data.utils import load_examples, format_prompt
from openai import OpenAI

# Baseline: base Mistral 7B, 500 test examples, notebooks/02_baseline_evaluation.ipynb
BASELINE_BERTSCORE_F1 = 0.7148823738098145

TEST_FILE = "/Users/harthikmallichetty/Desktop/code-sentinel-data-source/ref-test.jsonl"
MERGED_MODEL_PATH = "gs://code-sentinel-training-us/merged-model/run1"

def evaluate_fine_tuned_models(model_path, examples):
    """
    evaluate_fine_tuned_models(model_path, examples)

    This function takes in path to model merged with new training-adjusted weights and runs the model on examples to generate list of predictions.
    """

    model, tokenizer = load(model_path)

    predictions = []

    for line in examples:
        predictions.append(generate(model, tokenizer, format_prompt(line), max_tokens=256))

    return predictions

def evaluate_gpt4o_mini(examples):
    """
    evaluate_gpt4o_mini(examples)

    This function calls for Open AI API connection to run given examples on GPT-4o-mini to compare with our fine tuned model and baseline Mistral 7B model.
    """

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    predictions = []

    for line in examples:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": format_prompt(line)}],
            max_tokens=256
        )

        predictions.append(response.choices[0].message.content)

    return predictions

def compute_bertscore(predictions, references):
    """
    compute_bertscore(predictions, references)

    This function takes two lists of strings and returns a F1 score.
    """

    P, R, F1 = score(predictions, references, lang="en", model_type="distilbert-base-uncased")

    return F1.mean().item()

if __name__ == "__main__":
    examples = load_examples(TEST_FILE, 500)
    references = [example["comment"] for example in examples]

    fine_tuned_preds = evaluate_fine_tuned_models(MERGED_MODEL_PATH, examples)
    gpt4o_preds = evaluate_gpt4o_mini(examples)

    finetuned_f1 = compute_bertscore(fine_tuned_preds, references)
    gpt4o_f1 = compute_bertscore(gpt4o_preds, references)

    print("\n=== Results ===")
    print(f"Base Mistral 7B:      {BASELINE_BERTSCORE_F1:.4f}")
    print(f"Fine-tuned (run1):    {finetuned_f1:.4f}")
    print(f"GPT-4o-mini:          {gpt4o_f1:.4f}")