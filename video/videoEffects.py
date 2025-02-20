import cv2
import numpy as np
import logging

net = cv2.dnn.readNet("video/yolov3.weights", "video/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("video/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


def apply_effect(frame, effect_name, trigger=False):
    """Applies an effect to the frame based on the effect name.
    This is a response that comes from the interpretation of a
    prompt sent to OpenAI. We will need to verify that the
    effect name is valid before applying the effect name returned
    is deterministic"""
    logger = logging.getLogger(__name__)
    logger.info(f"Applying effect: {effect_name}")

    if effect_name == "water_color":
        logger.debug("Applying water color effect")
        return apply_water_color_effect(frame)
    elif effect_name == "heat_map":
        logger.debug("Applying heat map effect")
        return apply_heat_map_effect(frame)
    elif effect_name == "grayscale":
        logger.debug("Applying grayscale effect")
        return apply_grayscale_effect(frame)
    elif effect_name == "object_detection":
        return apply_object_detection_theme(frame, trigger)
    else:
        logger.debug("Applying default effect")
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


def apply_object_detection_theme(frame, button_trigger=False):
    height, width, channels = frame.shape

    # Detect objects
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
    )
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to store detected objects
    class_ids = []
    confidences = []
    boxes = []

    # Process detected objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to eliminate redundant overlapping boxes with lower confidences
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = []
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, f"{label} {round(confidence, 2)}", (x, y - 10), font, 1, color, 2
            )
            detected_objects.append((label, confidence, (x, y, w, h)))

    # If the button is triggered, save the detected objects
    if button_trigger:
        # Here you can save detected_objects to a file or database
        print("Detected objects:", detected_objects)

    return frame, detected_objects
