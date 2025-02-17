import cv2
from video.videoEffects import (
    apply_grayscale_effect,
    apply_water_color_effect,
    apply_heat_map_effect,
    apply_default_effect,
    apply_object_detection_theme,
)
import time


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Dictionary of effects and their corresponding functions
    effects = {
        "normal": apply_default_effect,
        "grayscale": apply_grayscale_effect,
        "water_color": apply_water_color_effect,
        "heat_map": apply_heat_map_effect,
        "object_detection": apply_object_detection_theme,
        # Add more effects here as you create them
    }

    current_effect = "normal"

    print("Video Effects Demo")
    print("Press 'q' to quit")
    print("Press 'n' for next effect")
    print("Note: Make sure the video window is in focus when pressing keys")

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Apply current effect
        if current_effect == "object_detection":
            processed_frame, _ = effects[current_effect](frame)  # Unpack only the frame
        else:
            processed_frame = effects[current_effect](frame)

        # Display effect name on frame
        cv2.putText(
            processed_frame,
            f"Effect: {current_effect}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Show the frame
        cv2.imshow("Video Effects Demo", processed_frame)

        # Handle keyboard input with a small sleep to prevent too rapid cycling
        key = cv2.waitKey(1) & 0xFF
        time.sleep(0.01)  # Add a small delay to prevent too rapid cycling
        if key == ord("q"):
            print("Quitting...")
            break
        elif key == ord("n"):
            # Cycle through effects
            effect_list = list(effects.keys())
            current_index = effect_list.index(current_effect)
            current_effect = effect_list[(current_index + 1) % len(effect_list)]
            print(f"Switching to effect: {current_effect}")

    # Clean up
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
