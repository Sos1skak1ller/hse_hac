import os
import pandas as pd
from typing import List
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

from app.utils.submit import generate_submit

load_dotenv()

solutions_df = pd.read_csv("~/hse_hac/data/raw/train/solutions.csv")
tasks_df = pd.read_csv("~/hse_hac/data/raw/train/tasks.csv")

model_name = "IlyaGusev/saiga_llama3_8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

def create_few_shot_context(num_examples: int = 5) -> str:
    examples = []
    for _, row in solutions_df.sample(n=num_examples).iterrows():
        task_id = row["task_id"]
        task_description = tasks_df[tasks_df["id"] == task_id]["task_description"].values[0]
        student_solution = row["student_solution"]
        author_comment = row["author_comment"]
        example = f"Task: {task_description}\nStudent's solution:\n{student_solution}\nTeacher's comment:\n{author_comment}\n"
        examples.append(example)
    return "\n\n".join(examples)

few_shot_context = create_few_shot_context()

def predict(row: pd.Series) -> str:
    # Add few-shot context to the input text
    input_text = f"{few_shot_context}\n\nTask: {row['task_description']}\nStudent's solution:\n{row['student_solution']}\nTeacher's comment:"
    try:
        output = model_pipeline(input_text, max_length=200, num_return_sequences=1)[0]["generated_text"]
        generated_comment = output.split("Teacher's comment:")[-1].strip()
        return generated_comment
    except Exception as e:
        print(f"Error while predicting: {e}")
        return ""

generate_submit(
    test_solutions_path="~/hse_hac/data/raw/test/solutions.csv",
    predict_func=predict,
    save_path="~/hse_hac/data/processed/submission.csv",
    use_tqdm=True,
)
