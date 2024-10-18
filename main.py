import os
import pandas as pd
from typing import List
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import gdown

from app.utils.submit import generate_submit

load_dotenv()

# # Указываем ссылки на файлы Google Диска
# train_solutions_url = 'https://docs.google.com/file/d/1wSKxoYUbXyhVADfCn8_I3wZAcELV0nD1/edit?usp=docslist_api&filetype=msexcel'
# train_tasks_url = 'https://docs.google.com/file/d/18aIRgmu6JjmVJV1ynm8Tlo8Cn7rqogRi/edit?usp=docslist_api&filetype=msexcel'
# test_solutions_url = 'https://docs.google.com/file/d/1fAl5vjnp9nG5GmZzpvI8i6vyTOKuHpTQ/edit?usp=docslist_api&filetype=msexcel'

# # Указываем пути для сохранения загружаемых файлов
# train_solutions_path = "../data/raw/train/solutions.xlsx"
# train_tasks_path = "../data/raw/train/tasks.xlsx"
# test_solutions_path = "../data/raw/test/solutions.xlsx"

# # Создаем директории, если они не существуют
# os.makedirs(os.path.dirname(train_solutions_path), exist_ok=True)
# os.makedirs(os.path.dirname(train_tasks_path), exist_ok=True)
# os.makedirs(os.path.dirname(test_solutions_path), exist_ok=True)

# # Скачиваем файлы с Google Диска
# gdown.download(train_solutions_url, train_solutions_path, quiet=False)
# gdown.download(train_tasks_url, train_tasks_path, quiet=False)
# gdown.download(test_solutions_url, test_solutions_path, quiet=False)

# # Загружаем данные
# solutions_df = pd.read_excel(train_solutions_path)
# tasks_df = pd.read_excel(train_tasks_path)

# Инициализация модели и токенизатора
model_name = "IlyaGusev/saiga_llama3_8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Создание пайплайна генерации текста
model_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# def create_few_shot_context(num_examples: int = 5) -> str:
#     # Создание контекста few-shot из случайных примеров
#     examples = []
#     for _, row in solutions_df.sample(n=num_examples).iterrows():
#         task_id = row["task_id"]
#         task_description = tasks_df[tasks_df["id"] == task_id]["task_description"].values[0]
#         student_solution = row["student_solution"]
#         author_comment = row["author_comment"]
#         example = f"Task: {task_description}\nStudent's solution:\n{student_solution}\nTeacher's comment:\n{author_comment}\n"
#         examples.append(example)
#     return "\n\n".join(examples)

# # Генерация контекста few-shot
# few_shot_context = create_few_shot_context()

# def predict(row: pd.Series, prompt: str) -> str:
#     # Используем переданный промпт перед основным текстом запроса
#     input_text = f"{prompt}\n\nTask: {row['task_description']}\nStudent's solution:\n{row['student_solution']}\nTeacher's comment:"
#     try:
#         output = model_pipeline(input_text, max_length=200, num_return_sequences=1)[0]["generated_text"]
#         generated_comment = output.split("Teacher's comment:")[-1].strip()
#         return generated_comment
#     except Exception as e:
#         print(f"Error while predicting: {e}")
#         return ""
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
    
# Пример использования простого запроса
input_text = "Вы опытный преподаватель, который обеспечивает конструктивную обратную связь. Проаналищируй вот этот код и выдй ответ discount  = float(input()) money = int(input()) print('Реализация проекта будет стоить {money} тыс. руб. без скидки. Со скидой стоимость составит {money- (money * discount)} тыс. руб.')" 
response = simple_request_to_model(input_text)
print("Response from the model:", response)

# Устанавливаем пользовательский промпт
custom_prompt = "Вы опытный преподаватель, который обеспечивает конструктивную обратную связь."

# Вызываем функцию generate_submit для генерации сабмита
generate_submit(
    test_solutions_path=test_solutions_path,
    predict_func=lambda row: predict(row, custom_prompt),
    save_path="../data/processed/submission.csv",
    use_tqdm=True,
)
