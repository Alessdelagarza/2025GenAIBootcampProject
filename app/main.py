import streamlit as st
import logging
import data.embeddings as llm
import video.videoEffects as fxs
import cv2

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    try:
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

        user_prompt = st.text_input("Enter a prompt to augment the video. Be creative!")
        video_run = st.toggle("Video Run", value=False)
        logger.info(f"Video run: {video_run}")

        if user_prompt:
            logger.info(f"Received user prompt: {user_prompt}")
            frame_name, similarity = llm.find_most_similar_frame(user_prompt)
            explanation = llm.explain_frame_selection(
                user_prompt, frame_name, similarity
            )
            st.write(
                f"AI Suggested Frame: {frame_name}\n" f"Description: {explanation}"
            )
        else:
            logger.info("No prompt received")
            frame_name = "normal"
            explanation = "Default effect applied"

        # Initialize video capture
        cap = cv2.VideoCapture(0)

        # Create a placeholder in the Streamlit app
        video_placeholder = st.empty()

        # Add a checkbox for object detection trigger
        checkbox_trigger = st.checkbox("Detect Objects", value=False)
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
                if detected_objects and detected_objects:
                    # Display detected objects
                    detected_objects_placeholder.write("Detected Objects:")
                    for obj, conf, _ in detected_objects:
                        detected_objects_placeholder.write(
                            f"- {obj}: {conf:.2f} confidence"
                        )
                    if checkbox_trigger:
                        video_run = False  # Stop the video only when trigger is active
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
