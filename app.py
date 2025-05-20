import cv2
import pickle
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image

from flask import Flask, request, jsonify
from flask_cors import CORS

from utils.pre_processing import faceDetect_resize_equalize
from utils.extract_uniform_lbp import extract_uniform_lbp
from utils.extract_hog import extract_hog_features
from utils.extract_face_landmarks import extract_face_landmarks
from utils.train_extract_pca import process_new_image, extract_pca_features
from utils.uploadImageToCloud import upload_image
from utils.connectdtb import get_mongo_client

db = get_mongo_client()
app = Flask(__name__)
CORS(app)

features_db = db['features'].find()

def find_similar_images(input_image, features_db, top_n=4):
  if isinstance(input_image, str):  
    input_img = faceDetect_resize_equalize(cv2.imread(input_image))
  else:  
    input_img = faceDetect_resize_equalize(input_image)
  
  if input_img is None:
    return []
  
  input_lbp = extract_uniform_lbp(input_img)

  with open('hog_svd_model.pkl', 'rb') as f:
    hog_svd_model = pickle.load(f)

  svd = hog_svd_model['svd']
  input_hog = svd.transform([extract_hog_features(input_img)])[0]
  input_landmarks = extract_face_landmarks(input_img)
  input_pca = extract_pca_features(process_new_image(input_img, (64,64)))
  
  input_combined = np.concatenate([
    (input_hog / np.linalg.norm(input_hog)) * (1 / np.sqrt(len(input_hog))),
    (input_lbp / np.linalg.norm(input_lbp)) * (1 / np.sqrt(len(input_lbp))),
    ((input_landmarks / np.linalg.norm(input_landmarks)) * 3.0) if np.linalg.norm(input_landmarks) > 0 else input_landmarks,
    (input_pca / np.linalg.norm(input_pca)) * (1 / np.sqrt(len(input_pca)))
  ])
  
  similarity_scores = []
  for item in features_db:
    similarity = np.dot(input_combined, item['features']) / (
      np.linalg.norm(input_combined) * np.linalg.norm(item['features']))
    similarity_scores.append((item['path'], similarity))
  
  similarity_scores.sort(key=lambda x: x[1], reverse=True)
  
  return similarity_scores[:top_n]

def image_to_base64(image_path):
  with open(image_path, "rb") as img_file:
    return base64.b64encode(img_file.read()).decode('utf-8')

@app.route('/AddData', methods=['POST'])
def api_upload(): 
  if 'images' not in request.files:
    return jsonify({'error': 'No images provided'}), 400
  
  files = request.files.getlist('images')
  if not files:
    return jsonify({'error': 'No images provided'}), 400
  
  features_db = []
  for file in files:
    filename = file.filename
    ext = os.path.splitext(filename)[1]
    file_bytes = file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    imgHandled = faceDetect_resize_equalize(img)

    width = 400
    height = int(img.shape[0] * (width / img.shape[1]))
    img_resized = cv2.resize(img, (width, height))
    url_cloud = upload_image(img_resized, ext)

    with open('hog_svd_model.pkl', 'rb') as f:
      hog_svd_model = pickle.load(f)
    svd = hog_svd_model['svd']
    weight_landmarks = 3.0 

    lbp_features = extract_uniform_lbp(imgHandled)
    hog_features = svd.transform([extract_hog_features(imgHandled)])[0]
    landmarks_features = extract_face_landmarks(imgHandled)
    pca_features = extract_pca_features(process_new_image(imgHandled, (64,64)))

    print("hog_features", hog_features.shape)
    print("lbp_features", lbp_features.shape)
    print("landmarks_features", landmarks_features.shape)
    print("pca_features", pca_features.shape)


    combined_features = np.concatenate([
			(hog_features / np.linalg.norm(hog_features)) * (1 / np.sqrt(len(hog_features))),
			(lbp_features / np.linalg.norm(lbp_features)) * (1 / np.sqrt(len(lbp_features))),
			((landmarks_features / np.linalg.norm(landmarks_features)) * weight_landmarks) if np.linalg.norm(landmarks_features) > 0 else landmarks_features,
			(pca_features / np.linalg.norm(pca_features)) * (1 / np.sqrt(len(pca_features)))
		])
    
    db['features'].insert_one({
      'path': url_cloud,
      'features': combined_features.tolist()
    })

  return jsonify({
    'code': 200,
    'message': 'Images uploaded successfully'
  })
  # results = []
  
  # for file in files:
  #   # Read and process the uploaded image
  #   file_bytes = file.read()
  #   nparr = np.frombuffer(file_bytes, np.uint8)
  #   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
  #   # Find similar images
  #   similar_images = find_similar_images(img, features_db, top_n=4)
    
  #   # Prepare response with base64 encoded images
  #   for i, (path, score) in enumerate(similar_images):
  #     results.append({
  #       'rank': i+1,
  #       'path': path,
  #       'score': float(score),
  #       'image_base64': image_to_base64(path)
  #     })
  
  return jsonify({'results': len(files)})


@app.route('/find_similar', methods=['POST'])
def api_find_similar():
  if 'image' not in request.files:
    return jsonify({'error': 'No image provided'}), 400
  
  file = request.files['image']
  
  # Read and process the uploaded image
  file_bytes = file.read()
  nparr = np.frombuffer(file_bytes, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  
  # Find similar images
  similar_images = find_similar_images(img, features_db, top_n=4)
  
  # Prepare response with base64 encoded images
  results = []
  for i, (path, score) in enumerate(similar_images):
    results.append({
      'rank': i+1,
      'path': path,
      'score': float(score)
    })
  
  return jsonify({'results': results})

if __name__ == '__main__':
  app.run(debug=True)