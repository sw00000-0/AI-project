# Session 01 — Python Basics for AI
**Phase:** Foundation | **Prereq:** Python installed, VS Code set up

---

## Pre-Class Reading (do before session)

| Resource | What to focus on | Time |
|----------|-----------------|------|
| [W3Schools Python Basics](https://www.w3schools.com/python/python_intro.asp) | Variables, lists, loops, functions — skim if already familiar | 30 min |
| [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html) | Arrays — what they are and why they matter | 20 min |
| [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/intro_tutorials/01_table_oriented.html) | Think of a DataFrame like an Excel sheet in code | 20 min |

**Key question to think about before class:**
> "If AI models learn from data, what does data actually look like in code?"

---

## In-Class Agenda

### 1. Why Python for AI?
- Python is the standard language for AI/ML — not because it's fastest, but because of its libraries
- The three core libraries: **NumPy** (numbers/math), **pandas** (tables/data), **sklearn** (ML models)
- Show the ecosystem picture: raw data → pandas → numpy → sklearn → prediction

### 2. NumPy
- What is an array vs a Python list — and why arrays are better for math
- Creating arrays: `np.array([1,2,3])`, `np.zeros()`, `np.ones()`, `np.arange()`
- Array operations: element-wise math, slicing, reshaping
- **Concept:** an image is just a 3D array (height × width × colour channels)

```python
import numpy as np

# Create a simple array
arr = np.array([10, 20, 30, 40, 50])
print(arr * 2)          # element-wise multiply
print(arr[1:4])         # slicing
print(arr.shape)        # shape of the array

# 2D array (like a small image patch)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(matrix.shape)     # (2, 3) — 2 rows, 3 columns
```

### 3. Pandas
- Loading a CSV: `pd.read_csv()`
- Exploring data: `.head()`, `.info()`, `.describe()`
- Selecting columns and rows: `df['column']`, `df[df['age'] > 25]`
- Handling missing values: `.isnull().sum()`, `.dropna()`, `.fillna()`
- Basic aggregations: `.groupby()`, `.value_counts()`

```python
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv')

print(df.head())              # first 5 rows
print(df.info())              # column types and nulls
print(df.describe())          # stats for numeric columns
print(df['age'].mean())       # average age
print(df['survived'].value_counts())  # survival count
```

### 4. sklearn
- The sklearn pattern: **load data → split → fit → predict → evaluate**
- Classification example: predict if a passenger survived (Titanic)
- Key functions: `train_test_split`, `DecisionTreeClassifier`, `accuracy_score`
- Don't worry about *how* the model works internally yet — focus on the *pipeline*

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv')
df = df[['survived', 'pclass', 'age', 'fare']].dropna()

X = df[['pclass', 'age', 'fare']]   # features (inputs)
y = df['survived']                   # label (output to predict)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

---

## Practice Problems

### Problem 1 — NumPy Warm-up
Open a new notebook (Colab or VS Code). Without copy-pasting the examples above:
1. Create an array of numbers 1–20
2. Select only even numbers using slicing or filtering
3. Compute the mean and standard deviation
4. Create a 4×4 matrix of zeros, then set the diagonal values to 1

### Problem 2 — Pandas Exploration
Load the Titanic dataset (link in code above) and answer these questions using code:
1. How many passengers are in the dataset?
2. What is the average fare paid?
3. How many passengers were in each class (pclass)?
4. What is the survival rate for male vs female passengers?
5. Which 3 columns have the most missing values?

### Problem 3 — Your First ML Pipeline
Using the Titanic dataset:
1. Choose at least 4 features (columns) that you think are useful for predicting survival
2. Drop rows with missing values in those columns
3. Split into 80% train / 20% test
4. Train a `DecisionTreeClassifier`
5. Print the accuracy
6. **Challenge:** Try `max_depth=3` in the classifier and compare accuracy
