import streamlit as st
import logging
import signal
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_background():
    """Sets up the cyan background for the Streamlit app and header."""
    st.markdown(
        """
        <style>
            .stApp {
                background-color: #D0FAFD;
            }
            header[data-testid="stHeader"] {
                background-color: #D0FAFD;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def setup_sidebar():
    """Sets up the white background for the sidebar."""
    st.markdown(
        """
            <style>
                [data-testid=stSidebar] {
                    background-color: #1BE1F2;
                }
            </style>
        """,
        unsafe_allow_html=True,
    )


def kill_app_button():
    """Kills the Streamlit app."""
    if st.sidebar.button("Stop Application", type="primary", use_container_width=True):
        logger.info("Stop button pressed - terminating application")
        os.kill(os.getpid(), signal.SIGTERM)


def setup_input_box():
    """Sets up the input box and submit button for the Streamlit app."""
    user_prompt = st.text_area(
        "Enter a prompt to augment the video. Be creative!",
        height=100,
        max_chars=None,
        key="prompt_input",
    )

    button_container = st.container()
    with button_container:
        _, right_col = st.columns([4, 1])
        with right_col:
            submit_button = st.button(
                "Submit", type="primary", use_container_width=True
            )
    logger.info(f"Submit button: {submit_button}")

    if user_prompt:
        logger.info(f"User prompt: {user_prompt}")
        return user_prompt, submit_button
    else:
        return None, submit_button


def setup_spinner():
    """Sets up the spinner for the Streamlit app."""
    st.markdown(
        """
                <style>
                    .stSpinner {
                        background-color: #f0f0f0;
                        padding: 20px;
                        border-radius: 10px;
                    }
                </style>
                """,
        unsafe_allow_html=True,
    )


def light_green_blob(bold_text, text):
    """Sets up the light green blob for the Streamlit app."""
    st.markdown(
        f"""
        <div style='
            background-color: #F9FFDB;
            padding: 15px;
            border-radius: 10px;
            margin: 0 auto 20px auto;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            text-align: left;
        '>
            <strong>{bold_text}:</strong> {text}<br>
        </div>
        """,
        unsafe_allow_html=True,
    )


def light_pink_blob(bold_text, text):
    """Sets up the light green blob for the Streamlit app."""
    st.markdown(
        f"""
        <div style='
            background-color: #FFDBDE;
            padding: 15px;
            border-radius: 10px;
            margin: 0 auto 20px auto;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            text-align: left;
        '>
            <strong>{bold_text}:</strong> {text}<br>
        </div>
        """,
        unsafe_allow_html=True,
    )


def break_loop_button():
    """Breaks the loop in the Streamlit app by adding a stop button."""
    if st.button("Stop", type="primary", use_container_width=True):
        st.session_state.break_loop = True
        return True
    return False


def initialize_story_state():
    """Initializes the story state in the Streamlit app."""
    if "story_generated" not in st.session_state:
        st.session_state.story_generated = False
