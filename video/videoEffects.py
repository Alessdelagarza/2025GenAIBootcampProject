import cv2


def apply_effect(frame, effect_name):
    """Applies an effect to the frame based on the effect name.
    This is a response that comes from the interpretation of a
    prompt sent to OpenAI. We will need to verify that the
    effect name is valid before applying the effect name returned
    is deterministic"""
    if effect_name == "water color":
        return apply_water_color_effect(frame)
    elif effect_name == "heat map":
        return apply_heat_map_effect(frame)
    elif effect_name == "grayscale":
        return apply_grayscale_effect(frame)
    else:
        return apply_default_effect(frame)


def apply_water_color_effect(frame):
    """Creates a hand-drawn watercolor effect on the frame."""
    return cv2.stylization(frame, sigma_s=60, sigma_r=0.07)


def apply_heat_map_effect(frame):
    """applies a color mapping effect to the frame that makes it seems as if we are
    measuring temperature."""
    return cv2.applyColorMap(frame, cv2.COLORMAP_JET)


def apply_grayscale_effect(frame):
    """Converts the frame to grayscale (black and white image)."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def apply_default_effect(frame):
    """Applies no effect, returning the original frame."""
    return frame
