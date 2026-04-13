# Session 04 — Hugging Face, Datasets & Benchmarks
**Phase:** Core AI | **Prereq:** Sessions 01–03 complete

---

## Pre-Class Reading (do before session)

| Resource | What to focus on | Time |
|----------|-----------------|------|
| [Hugging Face — What is it? (official intro)](https://huggingface.co/docs/hub/index) | Skim: understand it's a hub for models, datasets, spaces | 10 min |
| [HF NLP Course — Chapter 1](https://huggingface.co/learn/nlp-course/chapter1/1) | What are Transformers? Read sections 1–4, skip code for now | 25 min |
| [Hugging Face Pipelines Docs](https://huggingface.co/docs/transformers/main_classes/pipelines) | Skim the list of available pipeline tasks | 10 min |
| [What are benchmarks? (short blog)](https://towardsdatascience.com/evaluating-language-models-through-benchmarks-1e7feda9c01c) | Why we benchmark AI models — just the concept | 10 min |

---

## In-Class Agenda

### 1. The Hugging Face Ecosystem
- **Model Hub:** thousands of pre-trained models you can download and run
- **Datasets Hub:** datasets ready to load in one line of code
- **Spaces:** demo apps built by the community (great for inspiration)
- **Why this matters for you:** you don't need to train models from scratch — use what exists

Navigate together: huggingface.co → Models → filter by task → pick a sentiment analysis model. Show the model card (what it does, how it was trained, limitations).

### 2. The `pipeline` API — Easiest Way to Use a Model
The `pipeline` function lets you run a model in ~3 lines of code.

```python
# Install first (once): pip install transformers torch
from transformers import pipeline

# Text classification (sentiment)
classifier = pipeline("sentiment-analysis")
result = classifier("I absolutely loved this movie, it was fantastic!")
print(result)
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Try multiple texts at once
texts = [
    "This product broke after one day.",
    "Great quality, fast shipping!",
    "It's okay, nothing special."
]
results = classifier(texts)
for text, res in zip(texts, results):
    print(f"{res['label']} ({res['score']:.2f}): {text}")
```

Available pipeline tasks:
```python
# Question answering
qa = pipeline("question-answering")
qa(question="What is the capital of France?",
   context="France is a country in Europe. Its capital city is Paris.")

# Text generation
gen = pipeline("text-generation", model="gpt2")
gen("Once upon a time in a land far away,", max_length=60)

# Zero-shot classification (no training needed!)
clf = pipeline("zero-shot-classification")
clf("The match ended 3-1 and the crowd went wild",
    candidate_labels=["sports", "politics", "technology"])

# Translation
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
translator("Hello, how are you?")

# Summarization
summarizer = pipeline("summarization")
summarizer("Your long article text here...", max_length=60, min_length=20)
```

### 3. Loading Datasets from Hugging Face
```python
from datasets import load_dataset

# Load a famous NLP dataset
dataset = load_dataset("imdb")
print(dataset)
# DatasetDict with 'train' and 'test' splits

# Look at one example
print(dataset['train'][0])

# Convert to pandas if you prefer working that way
df = dataset['train'].to_pandas()
print(df.head())
print(df['label'].value_counts())
```

Using your own data:
```python
# Load from a local CSV
from datasets import Dataset
import pandas as pd

df = pd.read_csv('my_data.csv')
my_dataset = Dataset.from_pandas(df)
```

### 4. Understanding Benchmarks
Benchmarks = standardised tests for AI models, like exams.

Key NLP benchmarks:
| Benchmark | Tests | What it measures |
|-----------|-------|-----------------|
| GLUE | 9 tasks | General language understanding |
| SuperGLUE | Harder tasks | More complex reasoning |
| SQuAD | Reading comprehension | Question answering from passages |
| MMLU | 57 subjects | Broad knowledge across domains |
| HumanEval | Code problems | Code generation ability |

- Higher = better on leaderboard
- **Watch out:** a model that scores high on benchmarks might still fail on your specific task
- Always test on *your own data* in *your own use case*

Model cards on HuggingFace show benchmark scores — learn to read them.

### 5. Fine-tuning
- Pre-trained model: already knows language from massive training
- Fine-tuning: take that model and train it a little more on *your specific data*
- Analogy: a person who already knows how to read English — you just teach them the jargon of your domain
- We won't fine-tune today (needs GPU time), but you'll use a fine-tuned model

```python
# Many models on HuggingFace are already fine-tuned for specific tasks
# e.g., this model was fine-tuned for medical text classification
clf = pipeline("text-classification", model="medical-model-name-here")
```

---

## Practice Problems

### Problem 1 — Explore the Hub
Go to huggingface.co/models:
1. Search for a model that does **image classification**
2. Search for a model that does **text-to-speech**
3. Find a model that has more than 1 million downloads
4. For one model of your choice: read its model card and write 3 bullet points about it (what it does, what data it was trained on, any limitations mentioned)

### Problem 2 — Pipeline Sampler
In a Colab notebook, try at least 4 different pipeline tasks from the list in class. For each one:
1. Run it on 3 different inputs (texts, questions, etc.)
2. Look at the output format — what does the model return?
3. Note one case where the model got something wrong or seemed uncertain

### Problem 3 — Sentiment Analysis on Real Data
1. Load the `imdb` dataset from HuggingFace
2. Take 20 random reviews from the test set
3. Run them through a sentiment-analysis pipeline
4. Compare the pipeline's prediction to the actual label (`dataset['test']['label']`)
5. How accurate is it on your 20 samples? (Manual count is fine)

### Problem 4 — Zero-shot Classifier
Use the `zero-shot-classification` pipeline to classify 10 headlines from any news site today into categories you define yourself (e.g., `["tech", "health", "politics", "sports", "entertainment"]`).
1. Did it get them right?
2. Find one case where it was wrong and write why you think it was confused.

### Problem 5 — Benchmark Reading
Go to the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard):
1. Find the top 3 models as of today
2. What benchmarks are they measured on?
3. In your own words: why can't you just use the #1 model for everything?
