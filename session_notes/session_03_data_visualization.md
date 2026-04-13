# Session 03 — Data Visualization (matplotlib & seaborn)
**Phase:** Foundation | **Prereq:** Sessions 01–02 complete

---

## Pre-Class Reading (do before session)

| Resource | What to focus on | Time |
|----------|-----------------|------|
| [Matplotlib: Pyplot tutorial](https://matplotlib.org/stable/tutorials/pyplot.html) | Just the first half — line plots, bar charts, labels | 20 min |
| [Seaborn Introduction](https://seaborn.pydata.org/introduction.html) | Skim the examples in the gallery, identify 3 chart types you haven't seen | 15 min |
| [Storytelling with Data — free chapter](https://www.storytellingwithdata.com/blog/2017/8/9/what-is-exploratory-analysis) | What is exploratory data analysis? Why visualise before modelling? | 10 min |
| [Kaggle: Data Visualization Course — Intro lesson](https://www.kaggle.com/learn/data-visualization) | Interactive lesson, free, runs in browser | 20 min |

---

## In-Class Agenda

### 1. Why Visualise Data?
- EDA (Exploratory Data Analysis) = using charts to understand your data *before* building models
- Spot missing values, outliers, skewed distributions, correlations
- Anscombe's Quartet — 4 datasets with identical statistics but wildly different plots
- Rule of thumb: **plot first, model second**

### 2. Matplotlib Fundamentals
- Figure vs Axes: think of Figure as the canvas, Axes as the individual chart on it
- The `plt.subplots()` pattern — the professional way to make plots

```python
import matplotlib.pyplot as plt
import numpy as np

# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y, color='steelblue', linewidth=2, label='sin(x)')
ax.set_title('Sine Wave')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
plt.show()
```

Core chart types and when to use them:

| Chart type | Use when |
|-----------|---------|
| Line plot | Showing change over time |
| Bar chart | Comparing categories |
| Histogram | Showing distribution of a single numeric variable |
| Scatter plot | Showing relationship between two numeric variables |
| Box plot | Comparing distributions across groups |

```python
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv')

# Bar chart — survival by class
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 1. Bar chart
df.groupby('pclass')['survived'].mean().plot(kind='bar', ax=axes[0])
axes[0].set_title('Survival Rate by Class')

# 2. Histogram
df['age'].dropna().plot(kind='hist', bins=30, ax=axes[1])
axes[1].set_title('Age Distribution')

# 3. Scatter plot
axes[2].scatter(df['age'], df['fare'], alpha=0.3)
axes[2].set_title('Age vs Fare')

plt.tight_layout()
plt.show()
```

### 3. Seaborn — Prettier Charts, Less Code
- Seaborn is built on matplotlib — adds statistical plots and nicer defaults
- Always works best with a pandas DataFrame

```python
import seaborn as sns
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv')
sns.set_theme(style='whitegrid')

# Heatmap — correlations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Correlation heatmap
corr = df[['survived','pclass','age','fare','sibsp','parch']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[0][0])
axes[0][0].set_title('Correlation Matrix')

# Count plot (categorical)
sns.countplot(data=df, x='pclass', hue='survived', ax=axes[0][1])
axes[0][1].set_title('Survival Count by Class')

# Box plot
sns.boxplot(data=df, x='pclass', y='age', ax=axes[1][0])
axes[1][0].set_title('Age Distribution by Class')

# Violin plot
sns.violinplot(data=df, x='survived', y='fare', ax=axes[1][1])
axes[1][1].set_title('Fare Distribution by Survival')

plt.tight_layout()
plt.show()
```

Key seaborn plots:
| Plot | Function | Use case |
|------|----------|---------|
| Heatmap | `sns.heatmap()` | Correlation matrix, any 2D data |
| Count plot | `sns.countplot()` | How many in each category |
| Box plot | `sns.boxplot()` | Distribution summary + outliers |
| Violin plot | `sns.violinplot()` | Like boxplot but shows full distribution |
| Pair plot | `sns.pairplot()` | All combinations of variables at once |
| Distribution | `sns.histplot()` | Distribution with optional KDE curve |

### 4. Pair Plot — Your EDA Power Tool
```python
# One line — plots every variable against every other variable
cols = ['survived', 'pclass', 'age', 'fare']
sns.pairplot(df[cols].dropna(), hue='survived')
plt.show()
```
Use this early in any project to quickly see what correlates with your label.

### 5. Chart Design Principles
- Always include titles, axis labels, and units
- Avoid 3D charts (they distort perception)
- Use colour purposefully — don't just use rainbow for decoration
- Less is more: remove gridlines and borders that don't add information

---

## Practice Problems

### Problem 1 — Recreate and Explain
Using the Titanic dataset, create a single figure with 4 subplots:
1. Histogram of passenger ages
2. Bar chart of survival rate by gender
3. Box plot of fare by passenger class
4. Scatter plot of age vs fare, coloured by survival status

Add titles and axis labels to every subplot.

### Problem 2 — Heatmap Investigation
1. Load the Titanic dataset
2. Compute the correlation matrix for numeric columns
3. Plot it as a seaborn heatmap with annotations
4. Answer in a comment: which two variables are most positively correlated? Most negatively?

### Problem 3 — Tell a Story with a Chart
Pick any freely available dataset from [Kaggle Datasets](https://www.kaggle.com/datasets) that interests you (movies, music, sports, food — anything). Create one chart that tells an interesting story from that data. Add a title that is a sentence, not just a label (e.g., "Female passengers had nearly 3x higher survival rates than males" — not just "Survival by Gender").

### Problem 4 — Pair Plot Analysis
1. Run `sns.pairplot()` on the Titanic dataset using columns: `age`, `fare`, `pclass`, `survived`
2. Look at the resulting grid of plots
3. Write 3 observations — patterns or relationships you can see. Use complete sentences.

### Problem 5 — Broken Chart Fix
The following code has 3 bugs (style/correctness issues). Find and fix all of them:

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv')

plt.bar(df['pclass'], df['survived'])
plt.show()
```

Hint: think about what this is actually plotting vs what it *should* plot to make sense.
