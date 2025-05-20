import numpy as np
import cv2
import mediapipe as mp

def extract_face_landmarks(image):
  mp_face_mesh = mp.solutions.face_mesh
  face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
  
  rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  results = face_mesh.process(rgb_image)
  
  if not results.multi_face_landmarks:
    return np.zeros(10)  
  
  landmarks = results.multi_face_landmarks[0].landmark
  
  h, w = image.shape[:2]
  landmarks_array = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
  
  left_eye = landmarks_array[33]
  right_eye = landmarks_array[263]
  nose_tip = landmarks_array[1]
  mouth_left = landmarks_array[61]
  mouth_right = landmarks_array[291]
  
  chin = landmarks_array[152]       
  forehead = landmarks_array[10]    
  left_eyebrow = landmarks_array[107] 
  right_eyebrow = landmarks_array[336] 
  
  ratios = []
  
  eyes_distance = np.linalg.norm(left_eye - right_eye)
  face_width = np.linalg.norm(landmarks_array[234] - landmarks_array[454])
  ratios.append(eyes_distance / face_width)
  
  left_eye_nose_distance = np.linalg.norm(left_eye - nose_tip)
  face_height = np.linalg.norm(forehead - chin)
  ratios.append(left_eye_nose_distance / face_height)
  
  nose_mouth_distance = (np.linalg.norm(nose_tip - mouth_left) + np.linalg.norm(nose_tip - mouth_right)) / 2
  ratios.append(nose_mouth_distance / face_height)
  
  mouth_width = np.linalg.norm(mouth_left - mouth_right)
  ratios.append(mouth_width / face_width)
  
  right_eye_mouth_distance = np.linalg.norm(right_eye - mouth_right)
  ratios.append(right_eye_mouth_distance / face_height)

  forehead_nose_distance = np.linalg.norm(forehead - nose_tip)
  ratios.append(forehead_nose_distance / face_height)
  
  nose_chin_distance = np.linalg.norm(nose_tip - chin)
  ratios.append(nose_chin_distance / face_height)
  
  eyebrows_distance = np.linalg.norm(left_eyebrow - right_eyebrow)
  ratios.append(eyebrows_distance / face_width)
  
  ratios.append(face_height / face_width)
  
  left_eye_to_eyebrow = np.linalg.norm(left_eye - left_eyebrow)
  right_eye_to_eyebrow = np.linalg.norm(right_eye - right_eyebrow)
  eye_eyebrow_avg_distance = (left_eye_to_eyebrow + right_eye_to_eyebrow) / 2
  ratios.append(eye_eyebrow_avg_distance / face_height)
  
  return np.array(ratios)