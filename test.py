import cv2
import mediapipe as mp

# Khởi tạo mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Cấu hình mô hình phát hiện khuôn mặt
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,  # 0 cho khoảng cách gần, 1 cho xa hơn (đến 5m)
    min_detection_confidence=0.5
)

# Đọc ảnh
image = cv2.imread('./Data/1 (10).jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Phát hiện khuôn mặt
results = face_detection.process(image_rgb)

# Xử lý kết quả
if results.detections:
    for i, detection in enumerate(results.detections):
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        
        # Chuyển tọa độ tương đối thành tọa độ tuyệt đối
        x = int(bboxC.xmin * iw)
        y = int(bboxC.ymin * ih)
        w = int(bboxC.width * iw)
        h = int(bboxC.height * ih)
        
        # Cắt khuôn mặt
        face_img = image[y:y+h, x:x+w]
        cv2.imwrite(f'face_{i}.jpg', face_img)