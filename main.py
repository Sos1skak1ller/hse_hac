import os
import pandas as pd
from typing import List
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import gdown
import torch
import urllib.request

from app.utils.submit import generate_submit

load_dotenv()

torch.cuda.empty_cache()

train_solutions_url = 'https://drive.usercontent.google.com/u/0/uc?id=1wSKxoYUbXyhVADfCn8_I3wZAcELV0nD1&export=download'
train_tasks_url = 'https://drive.usercontent.google.com/u/0/uc?id=18aIRgmu6JjmVJV1ynm8Tlo8Cn7rqogRi&export=download'
test_solutions_url = 'https://drive.usercontent.google.com/u/0/uc?id=1fAl5vjnp9nG5GmZzpvI8i6vyTOKuHpTQ&export=download'

train_solutions = urllib.request.urlopen(train_solutions_url).read()
train_tasks = urllib.request.urlopen(train_tasks_url).read()
test_solutions = urllib.request.urlopen(test_solutions_url).read()

def download_file(url: str, save_path: str):
    response = urllib.request.urlopen(url)
    with open(save_path, "wb") as f:
        f.write(response.read())

download_file(train_solutions_url, "../data/raw/train/train_solutions.xlsx")
download_file(train_tasks_url, "../data/raw/train/train_tasks.xlsx")
download_file(test_solutions_url, "../data/raw/test/test_solutions.xlsx")

train_solutions_df = pd.read_excel("../data/raw/train/train_solutions.xlsx")
train_tasks_df = pd.read_excel("../data/raw/train/train_tasks.xlsx")
test_solutions_df = pd.read_excel("../data/raw/test/test_solutions.xlsx")
# solutions_df = pd.read_excel(train_solutions_path)
# tasks_df = pd.read_excel(train_tasks_path)

# Инициализация модели и токенизатора
model_name = "IlyaGusev/saiga_llama3_8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# # Инициализация модели и токенизатора с использованием CPU
model_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

def get_random_records_as_text(df):
    random_records = df.sample(n=15)
    random_records = random_records[['student_solution', 'author_comment']]
    fewshot = ''
    for index, row in random_records.iterrows():
        fewshot+= f'{row['student_solution']} => {row['author_comment']}\n'
    return fewshot

print(get_random_records_as_text(train_solutions_df))

def generate_comment(role: str, fewshot: str, row: pd.Series) -> str:
    student_solution = row['student_solution']
    input_text = f"{role}\n{fewshot}\n\n Student's solution:\n{student_solution}\nTeacher's comment:"
    
    try:
        output = model_pipeline(input_text, max_length=2000, num_return_sequences=1)[0]["generated_text"]
        generated_comment = output.split("Teacher's comment:")[-1].strip()
        return generated_comment
    except Exception as e:
        print(f"Error while generating comment: {e}")
        return ""

role = "Вы опытный преподаватель, который обеспечивает конструктивную обратную связь."
fewshot_examples = get_random_records_as_text(train_solutions_df)

# Generate submission file
generate_submit(
    test_solutions_path='../data/raw/test/test_solutions.xlsx',
    predict_func=lambda row: generate_comment(role, fewshot_examples, row),
    save_path="../data/processed/submission.csv",
    use_tqdm=True,
)
