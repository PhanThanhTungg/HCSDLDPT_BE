import numpy as np
import cv2
import mediapipe as mp

# Histogram equalization
# This function equalizes the histogram of a grayscale image
def histogram_equalization(img):
	flat = img.flatten()
	hist = [0] * 256
	for pixel in flat:
		hist[pixel] += 1
	cdf = [0] * 256
	cdf[0] = hist[0]
	for i in range(1, 256):
		cdf[i] = cdf[i-1] + hist[i]
	cdf_min = min([x for x in cdf if x > 0])
	total = flat.size
	lut = [0] * 256
	for i in range(256):
		lut[i] = round((cdf[i] - cdf_min) / (total - cdf_min) * 255)
		lut[i] = max(0, min(255, lut[i]))
	equalized = [lut[p] for p in flat]
	return np.array(equalized, dtype=np.uint8).reshape(img.shape)

# This function equalizes the histogram of a color image
def histogram_equalization_color(img):
  out = np.zeros_like(img)
  for c in range(3):
      out[..., c] = histogram_equalization(img[..., c])
  return out

# Resize image
def resize_image(img, new_width, new_height):
	h, w = img.shape[:2]
	if len(img.shape) == 3:
		c = img.shape[2]
		resized = np.zeros((new_height, new_width, c), dtype=img.dtype)
	else:
		resized = np.zeros((new_height, new_width), dtype=img.dtype)
	for i in range(new_height):
		for j in range(new_width):
			src_x = int(j * w / new_width)
			src_y = int(i * h / new_height)
			src_x = min(src_x, w - 1)
			src_y = min(src_y, h - 1)
			resized[i, j] = img[src_y, src_x]
	return resized

def bgr_to_rgb(img):
	return img[..., ::-1]


# Face detection && resize && histogram equalization
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
	model_selection=1,
	min_detection_confidence=0.5
)

def faceDetect_resize_equalize(image):
	image_rgb = bgr_to_rgb(image)
	results = face_detection.process(image_rgb)
	if results.detections:
		for i, detection in enumerate(results.detections):
			bboxC = detection.location_data.relative_bounding_box
			ih, iw, _ = image.shape
			x = int(bboxC.xmin * iw)
			y = int(bboxC.ymin * ih)
			w = int(bboxC.width * iw)
			h = int(bboxC.height * ih)
			face_img = image[y:y+h, x:x+w]
			try:
				face_img = resize_image(histogram_equalization_color(face_img), 400, 400)
			except Exception as e:
				print(f"Error processing face image: {e}")
				return None
			return face_img
	else:
		return None