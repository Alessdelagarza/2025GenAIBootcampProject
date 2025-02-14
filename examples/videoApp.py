"""Video application that applies effects
to video feed based on user prompts."""

import cv2
import logging
import sys
from pathlib import Path
import streamlit as st
from openai import AzureOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up paths
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
config_file_path = root_dir / "config.json"
config_module_path = root_dir / "config.py"

# Log debug information
logger.info("=== Debug Information ===")
logger.info(f"Current directory: {current_dir}")
logger.info(f"Root directory: {root_dir}")
logger.info(f"Config file exists: {config_file_path.exists()}")
logger.info(f"Config module exists: {config_module_path.exists()}")
logger.info(f"Python path: {sys.path}")

# Add the root directory to Python path
sys.path.insert(0, str(root_dir))

try:
    import config

    config_instance = config.Config()
    logger.info("Config loaded successfully!")
    logger.info(f"API type: {config_instance.api_type}")
except ImportError as e:
    logger.error(f"Error importing config module: {str(e)}")
    raise
except Exception as e:
    logger.error(f"Error loading config: {str(e)}")
    raise

config = config.Config()

# Initialize OpenAI API
AZURE_OPENAI_ENDPOINT = config.api_base
AZURE_OPENAI_KEY = config.api_key
API_TYPE = config.api_type
API_VERSION = config.api_version

print(f"Using OpenAI API type: {API_TYPE}")
print(f"Using OpenAI API version: {API_VERSION}")


# Define available effects
def apply_van_gogh_effect(frame):
    """Applies a stylized filter to the frame, reminiscent of Van Gogh's art
    style."""
    return cv2.stylization(frame, sigma_s=60, sigma_r=0.07)


def apply_melting_effect(frame):
    """Applies a color mapping to simulate a melting effect, creating a
    distorted appearance."""
    return cv2.applyColorMap(frame, cv2.COLORMAP_JET)


def apply_default_effect(frame):
    """Applies no effect, returning the original frame."""
    return frame


# Interpret the user's prompt using OpenAI
def interpret_prompt(prompt):
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=API_VERSION,
    )

    # Describe available effects with their descriptions
    effect_descriptions = {
        "van gogh": """Apply a stylized filter to the frame, reminiscent of Van
                    Gogh's art style.""",
        "melting": """Apply a color mapping to simulate a melting effect,
                   creating a distorted appearance.""",
    }

    # Create a detailed prompt for OpenAI to choose an effect
    openai_prompt = f"""Analyze the following user prompt: '{prompt}' and
    determine which effect makes the most sense to apply based on these
    descriptions. If none are suitable, respond with 'none'."""
    for effect_name, description in effect_descriptions.items():
        openai_prompt += f"- {effect_name}: {description}\n"

    # Call OpenAI to interpret the prompt using ChatCompletion
    response = client.chat.completions.create(
        model="gpt-4",  # Use the appropriate model for your API
        messages=[
            {
                "role": "system",
                "content": """You are an assistant that helps
             determine image effects. Keep responses short and concise.""",
            },
            {"role": "user", "content": openai_prompt},
        ],
        max_tokens=40,
        temperature=0.5,
    )

    # Extract the raw message returned by OpenAI
    raw_message = response.choices[0].message.content.strip()
    effect_name = raw_message.lower()

    # Return both the interpreted effect and the raw message
    # Check if effect exists in descriptions or is "none",
    # otherwise return "none"
    is_valid = effect_name in effect_descriptions or effect_name == "none"
    selected_effect = effect_name if is_valid else "none"
    return selected_effect, raw_message


# Apply the selected effect
def apply_effect(frame, effect_name):
    if effect_name == "van gogh":
        return apply_van_gogh_effect(frame)
    elif effect_name == "melting":
        return apply_melting_effect(frame)
    else:
        return apply_default_effect(frame)  # No effect


# Main application function
def main():
    st.title("Real-Time Video Augmentation with OpenAI")
    prompt = st.text_input("Enter an effect prompt:")
    run = st.checkbox("Run")

    # Interpret the prompt outside the loop
    if prompt:
        effect_name, raw_message = interpret_prompt(prompt)
    else:
        effect_name, raw_message = (
            "none",
            """Using default effect:
        no effect applied.""",
        )

    # Display the raw message from OpenAI
    st.write(f"OpenAI response: {raw_message}")

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Create a placeholder in the Streamlit app
    video_placeholder = st.empty()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        # Apply the interpreted effect
        frame = apply_effect(frame, effect_name)

        # Display the frame in the placeholder
        video_placeholder.image(frame, channels="BGR", use_container_width=True)

    # Release the video capture when not running
    cap.release()


if __name__ == "__main__":
    main()
