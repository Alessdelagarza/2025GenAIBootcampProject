import streamlit as st
import logging
import data.embeddings as llm
import video.videoEffects as fxs
import cv2
import os
import signal

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    try:
        # Set cyan background for entire app
        st.markdown(
            """
            <style>
                .stApp {
                    background-color: cyan;
                }
            </style>
        """,
            unsafe_allow_html=True,
        )

        logger.info("Starting main application")
        st.markdown(
            "<h1 style='text-align: center; white-space: nowrap;'>"
            "GenAI Bootcamp Project Demo 2025</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<h2 style='text-align: center;'>"
            "Real-Time Video Augmentation with Open-AI</h2>",
            unsafe_allow_html=True,
        )

        # Style the sidebar with white color
        st.markdown(
            """
            <style>
                [data-testid=stSidebar] {
                    background-color: white;
                }
            </style>
        """,
            unsafe_allow_html=True,
        )

        # Add red stop button in the sidebar
        if st.sidebar.button(
            "Stop Application", type="primary", use_container_width=True
        ):
            logger.info("Stop button pressed - terminating application")
            os.kill(os.getpid(), signal.SIGTERM)

        user_prompt = st.text_input("Enter a prompt to augment the video. Be creative!")
        video_run = st.toggle("Video Run", value=False)
        logger.info(f"Video run: {video_run}")

        if user_prompt:
            logger.info(f"Received user prompt: {user_prompt}")
            with st.spinner("AI is analyzing your prompt..."):
                frame_name = llm.ai_frame_selection(user_prompt)
                explanation = llm.ai_explanation(frame_name, user_prompt)
            st.markdown(
                f"""
                <div style='
                    background-color: #f0f0f0;
                    padding: 15px;
                    border-radius: 10px;
                    margin: 0 auto 20px auto;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                    text-align: center;
                '>
                    <strong>AI Suggested Frame:</strong> {frame_name}<br>
                    <strong>Description:</strong> {explanation}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            logger.info("No prompt received")
            frame_name = "normal"
            explanation = "Default effect applied"

        # Initialize video capture
        cap = cv2.VideoCapture(0)

        # Create a placeholder in the Streamlit app
        video_placeholder = st.empty()
        with st.spinner("AI frame evaluation..."):
            evaluation = llm.ai_evaluation(frame_name, explanation, user_prompt)
            st.markdown(
                f"""
                <div style='background-color: #f0f0f0; padding: 15px;
                border-radius: 10px; margin: 0 auto 20px auto;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                text-align: center;'>
                <strong>AI Evaluation (Was this a good suggestion?):</strong> {evaluation}
                </div>
                """,
                unsafe_allow_html=True,
            )
        # Add a checkbox for object detection trigger
        checkbox_trigger = (
            st.checkbox("Detect Objects", value=False)
            if frame_name == "object_detection"
            else False
        )
        logger.info(f"Object detection trigger: {checkbox_trigger}")
        detected_objects_placeholder = st.empty()

        while video_run:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture video")
                st.error("Failed to capture video")
                break

            # Apply the effect and convert from BGR to RGB for Streamlit
            if frame_name == "object_detection":
                frame, detected_objects = fxs.apply_effect(
                    frame, frame_name, trigger=checkbox_trigger
                )
                if detected_objects:
                    detected_objects_placeholder.write("Detected Objects:")
                    all_objects = []
                    for obj, conf, _ in detected_objects:
                        all_objects.append(f"- {obj}: {conf:.2f}")
                    detected_objects_placeholder.write("\n".join(all_objects))
                    if checkbox_trigger:
                        video_run = False
                        detected_objects_placeholder.write("Detected Objects:")
                        detected_objects_placeholder.write("\n".join(all_objects))
            else:
                frame = fxs.apply_effect(frame, frame_name)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, use_container_width=True)

        # Release the video capture when not running
        cap.release()

    except Exception as e:
        logger.error(f"Error in main application: {str(e)}")
        raise


if __name__ == "__main__":
    main()
