import os
import pandas as pd
from typing import List
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import gdown

from app.utils.submit import generate_submit

load_dotenv()

# Указываем ссылки на файлы Google Диска
train_solutions_url = 'https://drive.google.com/file/d/ваш_file_id_для_train_solutions/view?usp=sharing'
train_tasks_url = 'https://drive.google.com/file/d/ваш_file_id_для_train_tasks/view?usp=sharing'
test_solutions_url = 'https://docs.google.com/file/d/1fAl5vjnp9nG5GmZzpvI8i6vyTOKuHpTQ/edit?usp=docslist_api&filetype=msexcel'

# Указываем пути для сохранения загружаемых файлов
train_solutions_path = "../data/raw/train/solutions.xlsx"
train_tasks_path = "../data/raw/train/tasks.xlsx"
test_solutions_path = "../data/raw/test/solutions.xlsx"

# Создаем директории, если они не существуют
os.makedirs(os.path.dirname(train_solutions_path), exist_ok=True)
os.makedirs(os.path.dirname(train_tasks_path), exist_ok=True)
os.makedirs(os.path.dirname(test_solutions_path), exist_ok=True)

# Скачиваем файлы с Google Диска
gdown.download(train_solutions_url, train_solutions_path, quiet=False)
gdown.download(train_tasks_url, train_tasks_path, quiet=False)
gdown.download(test_solutions_url, test_solutions_path, quiet=False)

# Загружаем данные
solutions_df = pd.read_excel(train_solutions_path)
tasks_df = pd.read_excel(train_tasks_path)

# Инициализация модели и токенизатора
model_name = "IlyaGusev/saiga_llama3_8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Создание пайплайна генерации текста
model_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

def create_few_shot_context(num_examples: int = 5) -> str:
    # Создание контекста few-shot из случайных примеров
    examples = []
    for _, row in solutions_df.sample(n=num_examples).iterrows():
        task_id = row["task_id"]
        task_description = tasks_df[tasks_df["id"] == task_id]["task_description"].values[0]
        student_solution = row["student_solution"]
        author_comment = row["author_comment"]
        example = f"Task: {task_description}\nStudent's solution:\n{student_solution}\nTeacher's comment:\n{author_comment}\n"
        examples.append(example)
    return "\n\n".join(examples)

# Генерация контекста few-shot
few_shot_context = create_few_shot_context()

def predict(row: pd.Series, prompt: str) -> str:
    # Используем переданный промпт перед основным текстом запроса
    input_text = f"{prompt}\n\nTask: {row['task_description']}\nStudent's solution:\n{row['student_solution']}\nTeacher's comment:"
    try:
        output = model_pipeline(input_text, max_length=200, num_return_sequences=1)[0]["generated_text"]
        generated_comment = output.split("Teacher's comment:")[-1].strip()
        return generated_comment
    except Exception as e:
        print(f"Error while predicting: {e}")
        return ""

# Устанавливаем пользовательский промпт
custom_prompt = "Вы опытный преподаватель, который обеспечивает конструктивную обратную связь."

# Вызываем функцию generate_submit для генерации сабмита
generate_submit(
    test_solutions_path=test_solutions_path,
    predict_func=lambda row: predict(row, custom_prompt),
    save_path="../data/processed/submission.csv",
    use_tqdm=True,
)
