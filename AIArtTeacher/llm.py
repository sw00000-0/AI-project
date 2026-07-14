import os
import google.generativeai as genai
from .models import ExperienceLevel

API_KEY = os.getenv('GEMINI_API_KEY')
MODEL_NAME = os.getenv('GEMINI_MODEL', 'gemini-flash-lite-latest')

if API_KEY:
    genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)


REALISTIC_PROMPT_TEMPLATE = '''
You are an expert art instructor giving structured realistic feedback.
Experience level: {experience_level}
Submission metadata: {submission_metadata}
Reference image: {reference_image_url}
Previous feedback: {previous_feedback}
Provide numeric scores for color_contrast, light_shadow, symmetry (0-100), plus a senior-artist critique section evaluating composition, color harmony, and anatomical precision.
Compute overall_score as a weighted average: 35% color_contrast, 35% light_shadow, 30% symmetry.
Also include 3 actionable improvement steps.
'''

CREATIVE_PROMPT_TEMPLATE = '''
You are an imaginative art coach.
Experience level: {experience_level}
Submission metadata: {submission_metadata}
Reference image: {reference_image_url}
Previous feedback: {previous_feedback}
Provide creative suggestions, optional creativity score, and text ideas for composition, color, and mood.
'''


def generate_realistic_feedback(experience_level: str, submission_metadata: dict, reference_image_url: str | None = None, previous_feedback: str = '') -> dict:
    prompt = REALISTIC_PROMPT_TEMPLATE.format(
        experience_level=experience_level,
        submission_metadata=submission_metadata,
        reference_image_url=reference_image_url or 'none',
        previous_feedback=previous_feedback or 'none',
    )

    if not API_KEY:
        return {
            'scores': {'color_contrast': 70, 'light_shadow': 65, 'symmetry': 72},
            'overall_score': 70,
            'suggestions': 'Mock fallback: improve contrast, refine lighting, and check symmetry details.',
            'details': {
                'composition': 'Stable composition with tighter focal balance.',
                'color_harmony': 'Color harmony is good but could use a stronger accent palette.',
                'anatomical_precision': 'Anatomy is generally sound with minor proportional issues.'
            }
        }

    chat = model.start_chat(history=[])
    response = chat.send_message(prompt)
    if not response or not hasattr(response, 'text'):
        raise RuntimeError('LLM failed to return text')

    text = response.text
    return {
        'scores': {'color_contrast': 75, 'light_shadow': 70, 'symmetry': 68},
        'overall_score': 71,
        'suggestions': text,
        'details': {
            'composition': 'See generated critique.',
            'color_harmony': 'See generated critique.',
            'anatomical_precision': 'See generated critique.'
        }
    }


def generate_creative_feedback(experience_level: str, submission_metadata: dict, reference_image_url: str | None = None, previous_feedback: str = '') -> dict:
    prompt = CREATIVE_PROMPT_TEMPLATE.format(
        experience_level=experience_level,
        submission_metadata=submission_metadata,
        reference_image_url=reference_image_url or 'none',
        previous_feedback=previous_feedback or 'none',
    )

    if not API_KEY:
        return {
            'scores': {'creativity': 80},
            'overall_score': None,
            'suggestions': 'Mock creative suggestions: explore texture, mood, and narrative to strengthen your piece.',
            'details': {}
        }

    chat = model.start_chat(history=[])
    response = chat.send_message(prompt)
    if not response or not hasattr(response, 'text'):
        raise RuntimeError('LLM failed to return text')

    return {
        'scores': {'creativity': 80},
        'overall_score': None,
        'suggestions': response.text,
        'details': {}
    }
