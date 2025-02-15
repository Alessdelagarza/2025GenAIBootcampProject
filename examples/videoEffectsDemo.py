import cv2
from video.videoEffects import (
    apply_grayscale_effect,
    apply_water_color_effect,
    apply_heat_map_effect,
    apply_default_effect,
)


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Dictionary of effects and their corresponding functions
    effects = {
        "normal": apply_default_effect,
        "grayscale": apply_grayscale_effect,
        "water_color": apply_water_color_effect,
        "heat_map": apply_heat_map_effect,
        # Add more effects here as you create them
    }

    current_effect = "normal"

    print("Video Effects Demo")
    print("Press 'q' to quit")
    print("Press 'n' for next effect")

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Apply current effect
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

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            # Cycle through effects
            effect_list = list(effects.keys())
            current_index = effect_list.index(current_effect)
            current_effect = effect_list[(current_index + 1) % len(effect_list)]

    # Clean up
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
