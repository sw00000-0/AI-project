# Session 05 — Building a Chatbot with Streamlit
**Phase:** Core AI | **Prereq:** Sessions 01–04 complete

---

## Pre-Class Reading (do before session)

| Resource | What to focus on | Time |
|----------|-----------------|------|
| [Streamlit — Getting Started](https://docs.streamlit.io/get-started/tutorials/create-an-app) | Run the "create an app" tutorial yourself | 25 min |
| [Streamlit Chat Elements](https://docs.streamlit.io/develop/api-reference/chat) | Read `st.chat_message` and `st.chat_input` docs | 10 min |
| [OpenAI Python Quickstart](https://platform.openai.com/docs/quickstart) OR [Claude API Quickstart](https://docs.anthropic.com/en/api/getting-started) | Pick one API — read just the first page, understand the request/response format | 15 min |
| [What is an API? (plain English)](https://www.howtogeek.com/343877/what-is-an-api/) | If you haven't worked with APIs before | 10 min |

**Key question to think about before class:**
> "What is the difference between using an AI through a website (like Claude.ai) vs. through an API?"

---

## In-Class Agenda

### 1. What is Streamlit? (5 min)
- A Python library for building web apps — no HTML/CSS/JavaScript needed
- You write Python → Streamlit turns it into a browser UI
- Perfect for AI demos and prototypes
- Install: `pip install streamlit`
- Run: `streamlit run app.py` → opens in your browser automatically

### 2. Streamlit Fundamentals (20 min)
```python
# app.py
import streamlit as st

# Basic elements
st.title("My First App")
st.write("Hello, world!")

# Text input
name = st.text_input("What is your name?")
if name:
    st.write(f"Hello, {name}!")

# Slider
number = st.slider("Pick a number", 0, 100, 50)
st.write(f"You picked: {number}")

# Button
if st.button("Click me"):
    st.balloons()

# Sidebar
with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
```

Key Streamlit concepts:
| Concept | Explanation |
|---------|------------|
| `st.session_state` | Stores data across re-runs (like a user's message history) |
| Re-run | Streamlit re-runs the whole script on every interaction |
| Widget | Any interactive element (button, input, slider) |
| Layout | `st.columns()`, `st.sidebar`, `st.expander()` |

### 3. Using an LLM API (20 min)

**Option A — Using Claude (Anthropic)**
```python
import anthropic

client = anthropic.Anthropic(api_key="your-api-key-here")

message = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain machine learning in one paragraph."}
    ]
)
print(message.content[0].text)
```

**Option B — Using GPT (OpenAI)**
```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key-here")

response = client.chat.completions.create(
    model="gpt-4o-mini",  # cheaper model for testing
    messages=[
        {"role": "user", "content": "Explain machine learning in one paragraph."}
    ]
)
print(response.choices[0].message.content)
```

**Multi-turn conversations:** APIs don't remember previous messages — you send the full history every time.
```python
conversation_history = [
    {"role": "user", "content": "Hi, my name is Mansi"},
    {"role": "assistant", "content": "Hello Mansi! How can I help you?"},
    {"role": "user", "content": "What's my name?"}
]
# The model can now answer "Your name is Mansi" because we sent the history
```

### 4. Building the Chatbot — Step by Step (25 min)

```python
# chatbot.py
import streamlit as st
import anthropic  # or: from openai import OpenAI

st.title("My AI Chatbot")

# --- Store API client and message history across re-runs ---
client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display existing chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- Get new user input ---
if prompt := st.chat_input("Ask me anything..."):

    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Call the API with full history
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1024,
                messages=st.session_state.messages
            )
            reply = response.content[0].text
            st.write(reply)

    # Add assistant reply to history
    st.session_state.messages.append({"role": "assistant", "content": reply})
```

Storing API keys safely:
```toml
# .streamlit/secrets.toml  (NEVER commit this file to GitHub)
ANTHROPIC_API_KEY = "sk-ant-..."
```

### 5. Customising the Chatbot (5 min)
- Add a **system prompt** to give the bot a persona (customer service agent, study buddy, recipe assistant...)
- Add a **Clear conversation** button: `if st.button("Clear"): st.session_state.messages = []`
- Add a **model selector** in the sidebar
- Show token usage / cost estimate in the sidebar

---

## Practice Problems

### Problem 1 — Hello Streamlit
Build and run a simple Streamlit app that:
1. Has a title and a short description
2. Takes a user's name as text input
3. Has a dropdown to pick a mood (Happy, Sad, Excited, Tired)
4. Displays a personalised greeting based on both inputs

Run it with `streamlit run app.py` and take a screenshot.

### Problem 2 — API Test
(Set up your API key first — instructor will provide or use free tier)
In a plain Python script (not Streamlit yet):
1. Send 3 different messages to the LLM API
2. Print the responses
3. Then send a 2-turn conversation — ask something, then ask a follow-up question that only makes sense if the model remembers the first message

### Problem 3 — Build the Basic Chatbot
Follow the chatbot code from class and get it running. Then add:
1. A sidebar with a "Clear Conversation" button
2. A title that includes the model name
3. A system prompt that makes the bot respond only about AI topics (refuse off-topic questions)

### Problem 4 — Persona Chatbot
Build a variant of the chatbot where the user can choose a persona for the AI from a sidebar dropdown:
- "Study Buddy" — explains things simply, like to a student
- "Strict Tutor" — gives hints but never full answers
- "Creative Writer" — responds in a poetic, creative style

Each persona should be implemented as a different system prompt.

### Problem 5 — Reflection
Answer in a short paragraph (5–7 sentences):
> "What is `st.session_state` and why is it necessary for a chatbot? What would happen if you didn't use it?"

---

## Vocabulary Added This Session
- API (Application Programming Interface), API key, endpoint
- Request, response, payload
- Session state, re-run
- System prompt, conversation history
- Token limit, context window (in the API sense)
- Widget, sidebar, spinner
