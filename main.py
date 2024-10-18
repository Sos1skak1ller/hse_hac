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

f = open("../data/raw/train/train_solutions.xlsx", "wb") #свои пути пишем там угу
f.write(train_solutions)
f.close()
f = open("../data/raw/train/train_tasks.xlsx", "wb") #свои пути пишем там угу
f.write(train_tasks)
f.close()
f = open("../data/raw/test/test_solutions.xlsx", "wb") #свои пути пишем там угу
f.write(test_solutions)
f.close()

train_solutions_df = pd.read_excel("../data/raw/train/train_solutions.xlsx")
train_tasks_df = pd.read_excel("../data/raw/train/train_tasks.xlsx")
test_solutions_df = pd.read_excel("../data/raw/test/test_solutions.xlsx")
# solutions_df = pd.read_excel(train_solutions_path)
# tasks_df = pd.read_excel(train_tasks_path)

# Инициализация модели и токенизатора
model_name = "IlyaGusev/saiga_llama3_8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Инициализация модели и токенизатора с использованием CPU
model_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

def get_random_records_as_text(df):
    random_records = df.sample(n=15)
    random_records = random_records[['student_solution', 'author_comment']]
    fewshot = ''
    for index, row in random_records.iterrows():
        fewshot+= f'{row['student_solution']} => {row['author_comment']}\n'
    return fewshot


def generate_comment(role: str, fewshot: str, row: pd.Series) -> str:
    """
    Генерирует комментарий на основе роли, few-shot примеров и данных из строки датафрейма.

    :param role: Роль, которую будет выполнять модель.
    :param fewshot: Few-shot примеры для обучения модели.
    :param row: Строка из датафрейма с данными для анализа.
    :return: Сгенерированный комментарий от модели.
    """
    # Формируем текст запроса
    input_text = f"{role}\n{fewshot}\n\nTask: {row['task_description']}\nStudent's solution:\n{row['student_solution']}\nTeacher's comment:"
    
    try:
        output = model_pipeline(input_text, max_length=200, num_return_sequences=1)[0]["generated_text"]
        generated_comment = output.split("Teacher's comment:")[-1].strip()
        return generated_comment
    except Exception as e:
        print(f"Error while generating comment: {e}")
        return ""

role = "Вы опытный преподаватель, который обеспечивает конструктивную обратную связь."
fewshot_examples = get_random_records_as_text(train_solutions_df)
example_row = test_solutions_df.sample(n=1).iloc[0]  # Получаем одну случайную строку из датафрейма

generated_comment = generate_comment(role, fewshot_examples, example_row)

def simple_request_to_model(input_text: str) -> str:
    """
    Отправляет простой запрос к нейросети и возвращает ответ.

    :param input_text: Текстовый ввод, который будет отправлен в модель.
    :return: Ответ от модели.
    """
    try:
        output = model_pipeline(input_text, max_length=200, num_return_sequences=1)[0]["generated_text"]
        return output.strip()
    except Exception as e:
        print(f"Error while sending request: {e}")
        return ""

response = simple_request_to_model(generated_comment)
print("Response from the model:", response)


# # Вызываем функцию generate_submit для генерации сабмита
# generate_submit(
#     test_solutions_path='../data/raw/test/test_solutions.xlsx',
#     predict_func=lambda row: generate_comment(role, fewshot_examples, row),  # Updated to use the correct parameters
#     save_path="../data/processed/submission.csv",
#     use_tqdm=True,
# )
