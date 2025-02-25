import streamlit as st
import streamlit_effects as sfx
import data.embeddings as llm
import video.videoEffects as fxs
import ai.ai_requests as ai
import automations.parking as prk
import logging
import time
import cv2
import config
import asyncio

config = config.Config()


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    sfx.setup_background()

    st.markdown(
        "<h1 style='text-align: center;'>GenAI Bootcamp 2025</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='text-align: center;'>Real-Time Video Augmentation with Open-AI</h2>",
        unsafe_allow_html=True,
    )

    st.sidebar.title("App Controls")
    sfx.setup_sidebar()
    sfx.kill_app_button()

    user_prompt, submit_button = sfx.setup_input_box()
    frame_name = "normal"

    if submit_button:
        with st.container():
            sfx.setup_spinner()
            with st.spinner("🔍 AI is selecting a frame..."):
                frame_name = llm.ai_frame_selection(user_prompt)
                with st.spinner("✨ AI is generating explanation..."):
                    explanation = llm.ai_explanation(frame_name, user_prompt)
                    with st.spinner("🧪 AI is evaluating suggestion..."):
                        evaluation = llm.ai_evaluation(
                            frame_name, explanation, user_prompt
                        )

        sfx.light_green_blob("AI Suggested Frame", frame_name)
        sfx.light_green_blob("Description", explanation)
        sfx.light_green_blob("Evaluation (Is This a Good Fit?)", evaluation)

        video_placeholder = st.empty()
        countdown_placeholder = st.empty()
        detected_objects_placeholder = st.empty()
        start_time = None
        cap = cv2.VideoCapture(0)
        sfx.initialize_story_state()

        while submit_button:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture video")
                st.error("Failed to capture video")
                break

            if frame_name == "object_detection":
                frame, detected_objects = fxs.apply_effect(
                    frame, frame_name, trigger=True
                )

                if detected_objects:
                    if start_time is None:
                        start_time = time.time()

                    elapsed_time = time.time() - start_time
                    remaining_time = max(10 - int(elapsed_time), 0)
                    countdown_placeholder.write(
                        f"Generating story in: {remaining_time} seconds"
                    )

                    detected_objects_placeholder.write("Detected Objects:")
                    all_objects = []
                    for obj, conf, _ in detected_objects:
                        all_objects.append(f"- {obj}: {conf:.2f}")
                    detected_objects_placeholder.write("\n".join(all_objects))

                    if elapsed_time >= 10 and not st.session_state.story_generated:
                        sfx.setup_spinner()
                        with st.spinner("AI is generating a story..."):
                            story = ai.ai_story(all_objects)
                        sfx.light_pink_blob("AI Generated Story", story)
                        countdown_placeholder.empty()
                        st.session_state.story_generated = True
            if frame_name == "ocr":
                if "ocr_performed" not in st.session_state:
                    st.session_state.ocr_performed = False

                frame = fxs.apply_effect(frame, frame_name)
                if start_time is None:
                    start_time = time.time()

                elapsed_time = time.time() - start_time
                remaining_time = max(10 - int(elapsed_time), 0)
                countdown_placeholder.write(
                    f"Performing OCR in: {remaining_time} seconds"
                )

                if elapsed_time >= 10 and not st.session_state.ocr_performed:
                    sfx.setup_spinner()
                    try:
                        ticket_number, ticket_time, ticket_date = fxs.get_ocr_text(
                            frame, config.ocr_api_key
                        )
                        if not all([ticket_number, ticket_time, ticket_date]):
                            raise ValueError(
                                "Could not extract all required ticket information"
                            )

                        ocr_result = {
                            "Ticket Number": ticket_number,
                            "Time": ticket_time,
                            "Date": ticket_date,
                        }
                        sfx.light_pink_blob("OCR Results", str(ocr_result))

                        with st.spinner("Performing OCR analysis..."):
                            with st.spinner("🚗 🎫 validating parking"):

                                async def run_with_timeout():
                                    try:
                                        await asyncio.wait_for(
                                            prk.navigate_website(
                                                ticket_number, ticket_date, ticket_time
                                            ),
                                            timeout=300,  # 5 minute timeout
                                        )
                                    except asyncio.TimeoutError:
                                        raise TimeoutError(
                                            "Parking validation timed out after 5 minutes"
                                        )

                                asyncio.run(run_with_timeout())
                    except Exception as e:
                        st.error(f"Error processing ticket: {str(e)}")
                        return
                    with st.expander("View Validation Screenshot", expanded=True):
                        st.image(
                            "screenshot.png", caption="Parking Validation Screenshot"
                        )
                    countdown_placeholder.empty()
                    st.session_state.ocr_performed = True
            else:
                frame = fxs.apply_effect(frame, frame_name)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, use_container_width=True)


if __name__ == "__main__":
    main()
