## Описание

## Запуск

### В локальной среде

```bash
poetry install
pip install gdown
```


## Структура проекта

```
.
├── app
│   ├── __init__.py
│   └── utils    <------------------------ утилиты
│       ├── __init__.py
│       ├── metric.py <------------------------ ознакомьтесь с метрикой
│       └── submit.py <------------------------ здесь всё для генерации сабмита
├── data
│   ├── complete <------------------------ подготовленные данные, сабмиты
│   ├── processed <----------------------- промежуточный этап подготовки данных
│   └── raw <----------------------------- исходные данные
│       ├── submit_example.csv
│       ├── test
│       │   ├── solutions.xlsx
│       │   ├── tasks.xlsx
│       │   └── tests.xlsx
│       └── train
│           ├── solutions.xlsx
│           ├── tasks.xlsx
│           └── tests.xlsx
├── main.py 
├── poetry.lock
├── pyproject.toml
├── README.md
└── tests
    ├── test_correctness.py <------------------------ проверить на корректность сабмит
    └── test_embedding_generation.py <--------------- попробовать генерацию эмбеддингов и подсчёт метрики
```