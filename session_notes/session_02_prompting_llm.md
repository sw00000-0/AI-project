# Session 02 — Prompting Techniques & LLM Skills
**Phase:** Foundation | **Prereq:** Session 01 complete

---

## Pre-Class Reading (do before session)

| Resource | What to focus on | Time |
|----------|-----------------|------|
| [Prompt Engineering Guide — Basics](https://www.promptingguide.ai/introduction/basics) | Zero-shot, few-shot, chain-of-thought — just read the definitions | 20 min |
| [What is a Large Language Model? (Visual explanation)](https://ig.ft.com/generative-ai/) | FT interactive explainer — no account needed, just scroll | 10 min |
| [OpenAI Playground](https://platform.openai.com/playground) or [Claude.ai](https://claude.ai) | Just play with it — try asking it to write code or explain a concept | 15 min |

---

## In-Class Agenda

### 1. What is an LLM and Why Should You Care?
- LLM = Large Language Model. Examples: GPT-4, Claude, Gemini, Llama
- Trained on massive amounts of text — learned patterns of language + knowledge
- **Key mental model:** it's a very powerful autocomplete — the quality of its output depends on the quality of your input (the prompt)
- How this changes your workflow as a developer: you now have a coding partner available 24/7

### 2. The Anatomy of a Good Prompt 
A strong prompt typically has:
- **Role** — who/what should the model act as
- **Context** — relevant background it needs
- **Task** — what you actually want done
- **Format** — how you want the output structured
- **Constraints** — what to avoid

```
Role:    You are a Python data scientist.
Context: I have a pandas DataFrame with columns: name, age, score.
         Some rows have missing values in the 'score' column.
Task:    Write code to fill missing scores with the column mean.
Format:  Provide only the code, no explanation.
```

vs. the bad prompt:
```
fill missing values
```

### 3. Prompting Techniques

#### Zero-shot prompting
Give the task directly — no examples. Works for simple, well-defined tasks.
```
Translate "Hello, how are you?" into French.
```

#### Few-shot prompting
Give 2–3 examples before your actual question. Works when the format matters.
```
Classify the sentiment:
"The food was amazing!" → Positive
"Terrible experience, never coming back." → Negative
"It was okay, nothing special." → [model fills this in]
```

#### Chain-of-Thought (CoT)
Ask the model to think step by step before answering. Massively improves complex tasks.
```
Solve this problem step by step:
A train travels 120km in 2 hours. How long will 300km take at the same speed?
```

#### System prompts vs user prompts
- **System prompt:** sets the persona/context at the start of the conversation (persistent)
- **User prompt:** your actual message each turn

#### Iterative refinement
Prompting is a conversation — if the first answer isn't right, don't start over.
```
That's close, but can you make the function handle edge cases where the list is empty?
```

### 4. Using GitHub Copilot Effectively
- **Inline completions:** type a comment describing what you want, let Copilot write the code
- **Copilot Chat:** ask questions about your code, request explanations, ask for tests
- **Good comment → good code:** the comment is the prompt

```python
# Load a CSV file and return only rows where 'status' is 'active'
# Handle the case where the file doesn't exist
def load_active_records(filepath):
    ...  # Copilot will complete this
```

Demo: write a data cleaning function using only natural language comments.

### 5. Common Mistakes to Avoid
- Vague prompts ("make it better" — better how?)
- Assuming the model remembers previous chats (it doesn't, unless using memory features)
- Not specifying the output format → you get an essay when you wanted a list
- Trusting code output without reading it — always review generated code

---

## Practice Problems

### Problem 1 — Prompt Comparison
Write the same request in two ways:
1. A vague one-line prompt
2. A detailed structured prompt (role + context + task + format)

Test both in Claude.ai or ChatGPT. Screenshot the outputs. Which one gave a more useful response? Write 2 sentences explaining why.

**Topic:** "Explain how a for loop works in Python"

### Problem 2 — Few-shot Prompting
Create a few-shot prompt that classifies AI-related news headlines as:
- `Research` — about new AI models or papers
- `Product` — about an AI product or feature launch
- `Ethics` — about AI safety, regulation, or societal impact

Give 3 examples in your prompt, then test it on 5 new headlines (find real ones on news.ycombinator.com or techcrunch.com).

### Problem 3 — Copilot Coding Exercise
In VS Code with Copilot enabled:
1. Create a new Python file called `data_cleaner.py`
2. Write only comments (no code) describing a function that:
   - Takes a pandas DataFrame
   - Removes duplicate rows
   - Fills numeric NaN values with column medians
   - Returns the cleaned DataFrame
3. Let Copilot generate the code
4. Read the generated code — does it match what you asked for? Fix anything wrong.

### Problem 4 — Chain-of-Thought Debug
Use Claude/ChatGPT to debug this broken code. **Do not tell the model what the bug is** — ask it to think step by step and find the issue itself.

```python
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total / len(numbers)

result = calculate_average([])
print(result)
```

Show the prompt you used and the model's answer.
