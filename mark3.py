import cv2
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import json

# Load MoveNet
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

# Target Pose Definition (can be loaded from patient profile later)
TARGET_POSE = {
    "Left Arm": 90,
    "Right Arm": 90,
    "Left Shoulder": 120,
    "Right Shoulder": 120
}
TOLERANCE = 15  # Angle tolerance for match

def detect_pose(img):
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 192)
    input_img = tf.cast(img, dtype=tf.int32)
    outputs = movenet(input_img)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]
    return keypoints

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

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
            pt1, pt2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
            cv2.line(frame, pt1, pt2, color, 2)

def show_angle_box(frame, keypoints, a_idx, b_idx, c_idx, label, y_offset, angle_store, threshold=0.3):
    h, w, _ = frame.shape
    a, b, c = keypoints[a_idx], keypoints[b_idx], keypoints[c_idx]
    if a[2] > threshold and b[2] > threshold and c[2] > threshold:
        angle = calculate_angle(
            (a[1]*w, a[0]*h),
            (b[1]*w, b[0]*h),
            (c[1]*w, c[0]*h)
        )
        angle_store[label] = angle
        text = f"{label}: {int(angle)}°"
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (10, y_offset - size[1] - 5), (10 + size[0] + 10, y_offset + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def optimized_match_pose(current_pose, target_pose, tolerance):
    for key, target_angle in target_pose.items():
        current_angle = current_pose.get(key)
        if current_angle is None:
            return False
        if abs(current_angle - target_angle) > tolerance:
            return False
    return True

def save_report_to_json(pose_data):
    report = {
        "matched_pose": {k: int(v) for k, v in pose_data.items()}
    }
    with open("pose_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print("\n JSON report saved as 'pose_report.json'")

def process_frame(frame, report=False):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoints = detect_pose(img_rgb)
    keypoints = np.clip(keypoints, 0, 1)

    angle_data = {}
    draw_keypoints(frame, keypoints)

    pairs = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6)]
    draw_lines(frame, keypoints, pairs)

    # Angles to detect
    angles = [
        (5, 7, 9, "Left Arm"),
        (6, 8, 10, "Right Arm"),
        (11, 5, 7, "Left Shoulder"),
        (12, 6, 8, "Right Shoulder")
    ]
    for i, (a, b, c, label) in enumerate(angles):
        show_angle_box(frame, keypoints, a, b, c, label, 30 + i*30, angle_data)

    if report:
        cv2.imwrite("pose_report.png", frame)
        save_report_to_json(angle_data)
        print(" Pose Image saved as 'pose_report.png'")
        for k, v in angle_data.items():
            print(f"  {k}: {int(v)}°")

    return frame, angle_data

def main():
    cap = cv2.VideoCapture(0)
    report_generated = False

    print("Stay in the target pose. When it matches, a report will be generated.")
    print("Press 'q' to quit. Press 'r' to reset.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, angles = process_frame(frame.copy())

        if not report_generated and optimized_match_pose(angles, TARGET_POSE, TOLERANCE):
            processed_frame, _ = process_frame(frame.copy(), report=True)
            report_generated = True

        cv2.imshow("Pose Detector", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            report_generated = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
