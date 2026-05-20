import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import csv
import os
import shutil
import random



def split_landmarks(landmarks):
    points_list = []
    for i in range(len(landmarks[0])):
        points_list.append(np.array([landmarks[0][i].x, landmarks[0][i].y, landmarks[0][i].z]))
    return np.array(points_list)

def normalize_landmarks_rotation_invariant(X20: np.ndarray) -> np.ndarray:
    # 1. Translation (poignet à l'origine)
    wrist = X20[0]
    Xc = X20 - wrist

    # 2. Définir axes de la main
    v1 = Xc[5]   # direction index
    v2 = Xc[17]  # direction auriculaire

    # Axe X
    x_axis = v1 / (np.linalg.norm(v1) + 1e-8)

    # Axe Z (perpendiculaire au plan de la paume)
    z_axis = np.cross(v1, v2)
    z_axis /= (np.linalg.norm(z_axis) + 1e-8)

    # Axe Y (orthogonal aux deux autres)
    y_axis = np.cross(z_axis, x_axis)

    # 3. Matrice de rotation (changement de base)
    R = np.stack([x_axis, y_axis, z_axis], axis=1)  # (3,3)

    # 4. Appliquer la rotation
    Xr = Xc @ R

    # 5. Normalisation d’échelle
    scale = np.linalg.norm(Xr[8]) + 1e-8
    Xn = Xr / scale

    return Xn

model_path = "hand_landmarker.task"
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)              # Palm connections
]

def to_pixel(x_norm: float, y_norm: float, w: int, h: int) -> tuple[int, int]:
    # keep to [0,1] to avoid occasional out-of-range artifacts
    x = min(max(x_norm, 0.0), 1.0)
    y = min(max(y_norm, 0.0), 1.0)
    return int(x * w), int(y * h)

def draw_hand_landmarks_tasks_only(
    image_bgr: np.ndarray,
    hand_landmarks_list,
    connections=HAND_CONNECTIONS,
    draw_points=True,
    draw_connections=True,
    point_radius=3,
    point_thickness=-1,
    line_thickness=2,
):
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    for hand_landmarks in hand_landmarks_list:
        # Convert normalized landmarks to pixel coords
        pts = [to_pixel(lm.x, lm.y, w, h) for lm in hand_landmarks]

        if draw_connections:
            for a, b in connections:
                cv2.line(annotated, pts[a], pts[b], (0, 255, 0), line_thickness)

        if draw_points:
            for (x, y) in pts:
                cv2.circle(annotated, (x, y), point_radius, (0, 0, 255), point_thickness)

    return annotated

def run_hand_landmarker_on_image_tasks_only(image_path: str):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        running_mode=vision.RunningMode.IMAGE
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            print("Image not found:", image_path)
            return

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        return result.hand_landmarks
        #annotated = draw_hand_landmarks_tasks_only(cv_image, result.hand_landmarks)
        

    else:
        #print("No hands detected.")
        return 



### run_hand_landmarker_on_image_tasks_only("Dataset1\ASL_Dataset\Train\A\8.jpg")
def pipe_folder(folder, label, nb_images):
    counter = 0
    fail_counter = 0
    CSV_FILENAME = "_csv/Train/" + label +".csv"
    with open(CSV_FILENAME, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        images = [img for img in os.listdir(folder+label+"/")]
        sample_img = random.sample(images, nb_images)

        for filename in sample_img:
            image_path = os.path.join(folder + label, filename)
            counter += 1
            points = run_hand_landmarker_on_image_tasks_only(image_path)
            
            if points :
                split_points = split_landmarks(points)
                points_normalises = normalize_landmarks_rotation_invariant(split_points)
                writer.writerow(points_normalises)
            else :
                fail_counter += 1
        csv_file.flush()
        csv_file.close()
    print(label, "over")
    success = (counter-fail_counter)/counter
    print("success_rate ", success, " --- total :", counter, ", fails :", fail_counter)
    return(nb_images - fail_counter)

source = "DS_mix/Train/"

#dossiers = [d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d))]

IMG_COUNT = []
for d in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] :
    IMG_COUNT.append(pipe_folder(source, d, 7200))

print(IMG_COUNT)
print(min(IMG_COUNT))