import cv2
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

# Load MoveNet
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

def detect_pose(img):
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 192)
    input_img = tf.cast(img, dtype=tf.int32)
    outputs = movenet(input_img)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]
    return keypoints

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def draw_keypoints(frame, keypoints, threshold=0.3):
    h, w, _ = frame.shape
    for y, x, c in keypoints:
        if c > threshold:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

def draw_lines(frame, keypoints, pairs, color=(255, 0, 0), threshold=0.3):
    h, w, _ = frame.shape
    for a, b in pairs:
        y1, x1, c1 = keypoints[a]
        y2, x2, c2 = keypoints[b]
        if c1 > threshold and c2 > threshold:
            pt1 = (int(x1 * w), int(y1 * h))
            pt2 = (int(x2 * w), int(y2 * h))
            cv2.line(frame, pt1, pt2, color, 2)

def show_angle_box(frame, keypoints, a_idx, b_idx, c_idx, label, y_offset, threshold=0.3):
    h, w, _ = frame.shape
    a, b, c = keypoints[a_idx], keypoints[b_idx], keypoints[c_idx]
    if a[2] > threshold and b[2] > threshold and c[2] > threshold:
        angle = calculate_angle(
            (a[1] * w, a[0] * h),
            (b[1] * w, b[0] * h),
            (c[1] * w, c[0] * h)
        )
        text = f"{label}: {int(angle)}°"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_w, text_h = text_size
        cv2.rectangle(frame, (10, y_offset - text_h - 5), (10 + text_w + 10, y_offset + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def process_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoints = detect_pose(img_rgb)
    keypoints = np.clip(keypoints, 0, 1)

    draw_keypoints(frame, keypoints)

    # Upper body edges
    pairs = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12)]
    draw_lines(frame, keypoints, pairs)

    # Angles
    angles_to_show = [
        (5, 7, 9, "Left Arm"),
        (6, 8, 10, "Right Arm"),
        (11, 5, 7, "Left Shoulder"),
        (12, 6, 8, "Right Shoulder"),
        (11, 5, 6, "L→Shoulder→R"),
        (12, 6, 5, "R→Shoulder→L"),
        (11, 12, 0, "Spine (Hips→Head)"),
    ]
    for i, (a, b, c, label) in enumerate(angles_to_show):
        show_angle_box(frame, keypoints, a, b, c, label, 30 + i * 30)

    return frame

def main():
    cap = cv2.VideoCapture(0)
    captured_frame = None

    print("Press 'c' to capture a pose and view angles.")
    print("Press 'r' to return to live view.")
    print("Press 'q' to quit.")

    while True:
        if captured_frame is None:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Live Feed - Press 'c' to capture", frame)
        else:
            processed = process_frame(captured_frame.copy())
            cv2.imshow("Captured Pose with Angles - Press 'r' to resume", processed)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            ret, captured_frame = cap.read()
        elif key == ord('r'):
            captured_frame = None

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
