import streamlit as st
import logging
import data.embeddings as llm
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

        # Initialize video capture
        cap = cv2.VideoCapture(0)

        # Create a placeholder in the Streamlit app
        video_placeholder = st.empty()

        while video_run:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture video")
                st.error("Failed to capture video")
                break

            # Display the frame in the placeholder
            video_placeholder.image(frame, channels="BGR", use_container_width=True)

        # Release the video capture when not running
        cap.release()

    except Exception as e:
        logger.error(f"Error in main application: {str(e)}")
        raise


if __name__ == "__main__":
    main()
