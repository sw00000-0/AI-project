import os
import json
import requests
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


def set_page_style(dark: bool):
	bg = "#ffc0cb"  # pink
	text = "#000000"
	if dark:
		bg = "#2b0b0b"
		text = "#ffffff"
	css = f"""
	<style>
	.stApp {{
	  background: {bg};
	  color: {text};
	}}
	.streamlit-expanderHeader {{ color: {text}; }}
	textarea, .stTextInput>div>div>input {{ color: {text}; background: transparent; }}
	</style>
	"""
	st.markdown(css, unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def gemini_response(prompt: str) -> str:
	key = os.getenv("GEMINI_API_KEY")
	endpoint = os.getenv("GEMINI_ENDPOINT")
	model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
	if key and not endpoint:
		endpoint = f"https://api.gemini.google/v1/models/{model}:generate"
	if not key:
		# Free local chatbot fallback when no API key is missing
		normalized = prompt.strip().lower()
		if any(word in normalized for word in ["hello", "hi", "hey"]):
			return "Hello! I'm your free chatbot. How can I help you today?"
		if "help" in normalized:
			return "Ask me anything — I can answer simple questions, chat, and help you test the interface."
		if "weather" in normalized:
			return "I can't fetch live weather without an API, but I can still chat with you!"
		return f"Free chatbot response: {prompt}"
	headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
	payload = {"prompt": prompt}
	try:
		r = requests.post(endpoint, headers=headers, json=payload, timeout=30)
		r.raise_for_status()
		data = r.json()
		# Try common response shapes
		if isinstance(data, dict):
			if "output" in data and isinstance(data["output"], dict) and "text" in data["output"]:
				return data["output"]["text"]
			if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
				first = data["choices"][0]
				if isinstance(first, dict):
					return first.get("text") or first.get("message", {}).get("content", "")
		return json.dumps(data)
	except Exception as e:
		return f"Error calling Gemini API: {e}"


def main():
	st.set_page_config(page_title="Gemini Chatbot", layout="wide")

	if "messages" not in st.session_state:
		st.session_state.messages = []
	if "dark" not in st.session_state:
		st.session_state.dark = False

	st.sidebar.title("Controls")
	if st.sidebar.button("New Chat"):
		st.session_state.messages = []
	if st.sidebar.button("Clear Cache"):
		try:
			st.cache_data.clear()
			st.sidebar.success("Cache cleared")
		except Exception:
			st.sidebar.error("Could not clear cache in this environment")

	st.session_state.dark = st.sidebar.checkbox("Dark mode", value=st.session_state.dark)

	set_page_style(st.session_state.dark)

	st.title("Gemini Chatbot")
	key = os.getenv("GEMINI_API_KEY")
	endpoint = os.getenv("GEMINI_ENDPOINT")
	model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
	free_mode = not key
	if free_mode:
		st.info("Free chatbot mode is active. No Gemini key required.")
	elif not endpoint:
		st.info(f"Using Gemini model: {model} via default endpoint.")
		endpoint = f"https://api.gemini.google/v1/models/{model}:generate"
		st.markdown(f"Default endpoint: {endpoint}")

	col1, col2 = st.columns([3, 1])

	with col1:
		for i, msg in enumerate(reversed(st.session_state.messages)):
			role = msg.get("role", "user")
			text = msg.get("text", "")
			if role == "user":
				st.markdown(f"**You:** {text}")
			else:
				st.markdown(f"**Bot:** {text}")

		user_input = st.text_area("Message", key="input_area", height=120)
		if st.button("Send") and user_input.strip():
			st.session_state.messages.append({"role": "user", "text": user_input})
			with st.spinner("Thinking..."):
				resp = gemini_response(user_input)
			st.session_state.messages.append({"role": "bot", "text": resp})
			st.session_state.input_area = ""
			if resp.strip():
				st.success("Response generated successfully.")
			else:
				st.error("No response was generated. Try again.")

	with col2:
		st.markdown("**Session**")
		st.markdown(f"Messages: {len(st.session_state.messages)}")
		st.markdown("---")
		st.markdown("**Config (from .env)**")
		st.markdown(f"GEMINI_MODEL: {model}")
		st.markdown(f"GEMINI_ENDPOINT: {os.getenv('GEMINI_ENDPOINT') or 'Using default'}")


if __name__ == '__main__':
	main()

