import logging
import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
from werkzeug.utils import secure_filename

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Directories for saving results
UPLOAD_DIR = 'uploads'
CSV_DIR = 'results/csv'
JSON_DIR = 'results/json'
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

# Constants
K = 10
Hip_to_Shoulder_Ratio = 1.35
VALUE = {
    "Shoulder": .46,
    "SittingEye": 10.69,
    "Elbow": 3.38,
    "ElbowChair": 8.3,
    "KneeToFoot": 0.77,
    "BackToKnee": -11.85
}

# Initialize MediaPipe Pose
try:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    )
except Exception as e:
    logger.error(f"Failed to initialize MediaPipe: {str(e)}")
    raise

# Helper functions for calculations
def process_image(image):
    """Process an image using MediaPipe Pose to extract keypoints."""
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        raise

def calculate_y(landmarks, keypoint1, keypoint2, image_width):
    """Calculate vertical distance between two keypoints."""
    x1, y1 = landmarks[keypoint1].x * image_width, landmarks[keypoint1].y * image_width
    x2, y2 = landmarks[keypoint2].x * image_width, landmarks[keypoint2].y * image_width
    return abs(y2 - y1)

def calculate_measurement(landmarks, keypoint1, keypoint2, image_width):
    """Calculate distance between two keypoints."""
    x1, y1 = landmarks[keypoint1].x * image_width, landmarks[keypoint1].y * image_width
    x2, y2 = landmarks[keypoint2].x * image_width, landmarks[keypoint2].y * image_width
    return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

def calculate_avg(landmarks, keypoint1, keypoint2, keypoint3, image_width):
    """Calculate average position between three keypoints."""
    x1, y1 = landmarks[keypoint1].x * image_width, landmarks[keypoint1].y * image_width
    x2a, y2a = landmarks[keypoint2].x * image_width, landmarks[keypoint2].y * image_width
    x2b, y2b = landmarks[keypoint3].x * image_width, landmarks[keypoint3].y * image_width
    x2 = (x2a + x2b) / 2.0
    y2 = (y2a + y2b) / 2.0
    return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

def calculate_KtF(landmarks, keypoint1, keypoint2, keypoint3, image_width):
    """Calculate Knee to Foot measurement."""
    x1, y1 = landmarks[keypoint1].x * image_width, landmarks[keypoint1].y * image_width
    x2a, y2a = landmarks[keypoint2].x * image_width, landmarks[keypoint2].y * image_width
    x2b, y2b = landmarks[keypoint3].x * image_width, landmarks[keypoint3].y * image_width
    x2 = (x2a + x2b) / 2.0
    y2 = min(y2a, y2b)
    return ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

def pixel_to_cm(pixel_distance, type_value, focal_length=26, sensor_width=8, distance=1980):
    """Convert pixel measurements to centimeters."""
    pixel_size = sensor_width / pixel_distance
    real_distance = (pixel_distance * distance) / (focal_length / (sensor_width / 1000))
    return real_distance / 10 + type_value

def chair(shoulder_width, torso_len):
    """Determine body shape and recommend chair backrest."""
    ratio = shoulder_width / torso_len
    if ratio > 1.5:
        body_shape = "Inverted Triangle"
        recommend_backrest = "Wide upper support with lumbar curvature."
    elif ratio < 0.8:
        body_shape = "Pear"
        recommend_backrest = "Narrow upper support and broad lower lumbar support."
    else:
        body_shape = "Rectangular"
        recommend_backrest = "Balanced support with emphasis on lumbar and mid-back."
    return body_shape, recommend_backrest

def process_measurements(sitting_image, standing_image):
    """Process images and calculate measurements."""
    try:
        image_width = sitting_image.shape[1]

        # Process images to get landmarks
        sitting_landmarks = process_image(sitting_image)
        standing_landmarks = process_image(standing_image)
        if not sitting_landmarks or not standing_landmarks:
            raise ValueError("Pose landmarks could not be detected.")

        # Calculate measurements
        sitting_eye_height = calculate_y(sitting_landmarks, mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value, mp_pose.PoseLandmark.RIGHT_HIP.value, image_width)
        sitting_eye_height_cm = pixel_to_cm(sitting_eye_height, VALUE["SittingEye"])

        shoulder_width = calculate_measurement(standing_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, image_width)
        shoulder_width_cm = pixel_to_cm(shoulder_width, VALUE["Shoulder"])

        elbow_len = calculate_avg(sitting_landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_PINKY.value, mp_pose.PoseLandmark.RIGHT_INDEX.value, image_width)
        elbow_len_cm = pixel_to_cm(elbow_len, VALUE["Elbow"])

        elbow_to_chair = calculate_y(sitting_landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_HIP.value, image_width)
        elbow_to_chair_cm = pixel_to_cm(elbow_to_chair, VALUE["ElbowChair"])

        knee_to_foot = calculate_KtF(sitting_landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_HEEL.value, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value, image_width)
        knee_to_foot_cm = pixel_to_cm(knee_to_foot, VALUE["KneeToFoot"])

        back_to_knee = calculate_measurement(sitting_landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value, image_width)
        back_to_knee_cm = pixel_to_cm(back_to_knee, VALUE["BackToKnee"])

        hip_width_cm = shoulder_width_cm * Hip_to_Shoulder_Ratio
        sitting_height_cm = sitting_eye_height_cm + K

        torso_len = calculate_y(sitting_landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_HIP.value, image_width)
        body_shape, recommend_backrest = chair(shoulder_width, torso_len)

        basic_measurements = {
            "Shoulder_Width": round(shoulder_width_cm, 2),
            "Sitting_Eye_Height": round(sitting_eye_height_cm, 2),
            "Sitting_Height": round(sitting_height_cm, 2),
            "Elbow_Len": round(elbow_len_cm, 2),
            "Elbow_To_Chair": round(elbow_to_chair_cm, 2),
            "Knee_To_Foot": round(knee_to_foot_cm, 2),
            "Back_To_Knee": round(back_to_knee_cm, 2),
            "Hip_Width": round(hip_width_cm, 2),
            "Body_Shape": body_shape
        }

        ergonomic_measurements = {
            "Desktop_Height": round(sitting_eye_height_cm + knee_to_foot_cm, 2),
            "Keyboard_Position": round(elbow_len_cm + 2.54, 2),
            "Desk_Height": round(knee_to_foot_cm + elbow_to_chair_cm, 2),
            "Chair_Height": round(knee_to_foot_cm, 2),
            "Chair_Length": round(back_to_knee_cm, 2),
            "Chair_Backrest_Width": round(shoulder_width_cm + 10, 2),
            "Chair_Width": round(hip_width_cm + 5, 2),
            "Chair_Type": recommend_backrest
        }

        return {
            "basic_measurements": basic_measurements,
            "ergonomic_recommendations": ergonomic_measurements
        }

    except Exception as e:
        logger.error(f"Measurement error: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to confirm the server is running."""
    return jsonify({'status': 'healthy'}), 200

@app.route('/analyze/ergonomics', methods=['POST'])
def analyze_ergonomics():
    """Endpoint for analyzing both sitting and standing images."""
    try:
        if 'sitting_image' not in request.files or 'standing_image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Both sitting and standing images are required'
            }), 400

        sitting_image_file = request.files['sitting_image']
        standing_image_file = request.files['standing_image']

        # Save images locally
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        sitting_path = os.path.join(UPLOAD_DIR, f'sitting_{timestamp}_{secure_filename(sitting_image_file.filename)}')
        standing_path = os.path.join(UPLOAD_DIR, f'standing_{timestamp}_{secure_filename(standing_image_file.filename)}')
        sitting_image_file.save(sitting_path)
        standing_image_file.save(standing_path)

        # Read and process images
        sitting_image = cv2.imread(sitting_path)
        standing_image = cv2.imread(standing_path)
        measurements = process_measurements(sitting_image, standing_image)

        # Save results locally
        json_filename = f'measurements_{timestamp}.json'
        csv_filename = f'measurements_{timestamp}.csv'
        json_path = os.path.join(JSON_DIR, json_filename)
        csv_path = os.path.join(CSV_DIR, csv_filename)

        # Save JSON
        with open(json_path, 'w') as json_file:
            json.dump(measurements, json_file, indent=4)

        # Save CSV
        with open(csv_path, 'w') as csv_file:
            csv_file.write('Measurement,Value\n')
            for key, value in measurements["basic_measurements"].items():
                csv_file.write(f'{key},{value}\n')
            csv_file.write('\nErgonomic Recommendations\n')
            for key, value in measurements["ergonomic_recommendations"].items():
                csv_file.write(f'{key},{value}\n')

        return jsonify({
            'success': True,
            'measurements': measurements,
            'json_file': json_path,
            'csv_file': csv_path
        })

    except Exception as e:
        logger.exception("Unexpected error during image analysis")
        return jsonify({
            'success': False,
            'error': 'Internal server error during image processing'
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
