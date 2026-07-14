# Session 09 — Multimodal Chatbot (Text + Image)
**Phase:** AI for Images | **Prereq:** Sessions 01–08 complete

---

## Pre-Class Reading (do before session)

| Resource | What to focus on | Time |
|----------|-----------------|------|
| [Claude Vision — official docs](https://docs.anthropic.com/en/docs/build-with-claude/vision) | How to send images to Claude via API | 15 min |
| [OpenAI Vision Guide](https://platform.openai.com/docs/guides/vision) | Alternative: GPT-4V image input format | 15 min |
| [Streamlit File Uploader Docs](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader) | How to accept image uploads in Streamlit | 10 min |
| [What is multimodal AI? (IBM explainer)](https://www.ibm.com/topics/multimodal-ai) | Big picture — what multimodal means | 10 min |

**Key question to think about before class:**
> "You already built a chatbot that understands text. What would it take to let users send an image and ask questions about it?"

---

## In-Class Agenda

### 1. What is Multimodal AI? (10 min)
- **Unimodal:** one type of input — text only, or image only
- **Multimodal:** multiple types of input — text + image, text + audio, etc.
- Examples:
  - Ask "what is in this photo?" → model sees the image + your text
  - Send a chart and ask "what is the trend?" → data analyst use case
  - Send a photo of food and ask "estimate the calories" → health app
  - Send a code screenshot and ask "explain this code" → developer tool
- Models: Claude (claude-opus-4-6), GPT-4o, Gemini 1.5, LLaVA (open source)

### 2. Sending Images to Claude API (20 min)
Claude accepts images as base64-encoded data in the message content.

```python
import anthropic
import base64
from pathlib import Path

client = anthropic.Anthropic(api_key="your-api-key")

def encode_image(image_path: str) -> tuple[str, str]:
    """Encode a local image to base64 and detect its media type."""
    path = Path(image_path)
    suffix_to_media = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                       '.png': 'image/png', '.gif': 'image/gif',
                       '.webp': 'image/webp'}
    media_type = suffix_to_media.get(path.suffix.lower(), 'image/jpeg')
    with open(path, 'rb') as f:
        data = base64.standard_b64encode(f.read()).decode('utf-8')
    return data, media_type

def ask_about_image(image_path: str, question: str) -> str:
    """Send an image + question to Claude and return the answer."""
    image_data, media_type = encode_image(image_path)

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]
    )
    return message.content[0].text

# Test it
answer = ask_about_image("my_photo.jpg", "What objects can you see in this image?")
print(answer)
```

You can also send an image by URL (no base64 needed):
```python
message = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=512,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
                }
            },
            {"type": "text", "text": "Describe this image in detail."}
        ]
    }]
)
print(message.content[0].text)
```

### 3. Handling Images in Streamlit (10 min)
```python
import streamlit as st
from PIL import Image
import io

# File uploader — accepts common image formats
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Max file size: 5MB"
)

if uploaded_file:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded image", use_container_width=True)

    # Convert to bytes for API
    img_bytes = uploaded_file.getvalue()    # raw bytes
    import base64
    img_b64 = base64.standard_b64encode(img_bytes).decode('utf-8')
    media_type = f"image/{uploaded_file.type.split('/')[-1]}"
```

### 4. Building the Full Multimodal Chatbot (25 min)

```python
# multimodal_chatbot.py
import streamlit as st
import anthropic
import base64

st.set_page_config(page_title="Vision Chatbot", page_icon="👁️")
st.title("Vision Chatbot — Ask About Any Image")

client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "current_image_b64" not in st.session_state:
    st.session_state.current_image_b64 = None
if "current_media_type" not in st.session_state:
    st.session_state.current_media_type = None

# Sidebar — image upload
with st.sidebar:
    st.header("Upload an Image")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        st.session_state.current_image = uploaded
        st.session_state.current_image_b64 = base64.standard_b64encode(
            uploaded.getvalue()).decode('utf-8')
        st.session_state.current_media_type = f"image/{uploaded.type.split('/')[-1]}"
        st.image(uploaded, caption="Current image", use_container_width=True)

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if isinstance(msg["content"], list):
            for part in msg["content"]:
                if part["type"] == "text":
                    st.write(part["text"])
        else:
            st.write(msg["content"])

# User input
if prompt := st.chat_input("Ask about the image or anything else..."):
    # Build the message content
    if st.session_state.current_image_b64:
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": st.session_state.current_media_type,
                    "data": st.session_state.current_image_b64,
                }
            },
            {"type": "text", "text": prompt}
        ]
    else:
        content = prompt

    # Display user message
    with st.chat_message("user"):
        if st.session_state.current_image:
            st.image(st.session_state.current_image, width=200)
        st.write(prompt)

    # Add to history and call API
    st.session_state.messages.append({"role": "user", "content": content})

    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1024,
                messages=st.session_state.messages
            )
            reply = response.content[0].text
            st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
```

### 5. Useful Multimodal Prompts to Try (5 min)
- "What is in this image? Be specific."
- "What text can you read in this image?"
- "Describe the mood or emotion this image conveys."
- "What are the main colours used? What do they suggest?"
- "If this were a product photo, write a marketing description."
- "What safety issues or hazards can you see in this image?"
- "Translate any text you see in this image."

---

## Practice Problems

### Problem 1 — Image Q&A Script
Write a standalone Python script (`image_qa.py`) that:
1. Accepts an image path and a question as command-line arguments
2. Sends both to Claude
3. Prints the answer

Test it on 3 different images with different types of questions.

### Problem 2 — Batch Image Describer
Write a script that:
1. Takes a folder of images as input
2. Sends each image to Claude with the prompt: "Describe this image in one sentence."
3. Saves the results to a CSV file with columns: `filename`, `description`

### Problem 3 — Build and Extend the Chatbot
Get the multimodal chatbot from class running. Then add at least two of these features:
- A "Quick Questions" sidebar with pre-set prompts (buttons that populate the chat input)
- A "Download Conversation" button that saves the chat history as a text file
- An image URL input option (in addition to file upload)
- Token usage display in the sidebar

### Problem 4 — Specialised Vision App
Build a focused Streamlit app around one use case. Choose one:
- **Recipe Finder:** user uploads a fridge photo → app suggests a recipe using those ingredients
- **Plant Identifier:** user uploads a plant photo → app identifies it and gives care tips
- **Chart Explainer:** user uploads a graph/chart → app explains the main takeaway in simple words
- **Text Extractor:** user uploads any photo with text → app extracts and transcribes the text

### Problem 5 — Limitation Testing
Upload 5 images designed to test the limits of the model. Try:
- A very low-resolution image
- An image with small text
- A heavily edited/filtered photo
- An image with something unusual or ambiguous
- An abstract image

For each, write: what you expected, what the model said, and whether it was accurate.

---

## Vocabulary Added This Session
- Multimodal, vision-language model (VLM)
- Base64 encoding
- Image URL vs base64 input
- File uploader (Streamlit)
- Media type (MIME type): image/jpeg, image/png
- OCR (Optical Character Recognition)
- Visual grounding, object detection (concepts)
