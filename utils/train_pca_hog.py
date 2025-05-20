import os
import cv2
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pickle

from extract_hog import extract_hog_features
def load_hogs(folder):
  hogs = []
  for filename in os.listdir(folder):
      path = os.path.join(folder, filename)
      img = cv2.imread(path)
      if img is not None:
          img = cv2.resize(img, (400,400))
          hog = extract_hog_features(img)
          hogs.append(hog)
  return hogs

def train_pca_hog(folder='Data'):
  X_hog = load_hogs(folder)
  print(f"Loaded HOG features: shape = {np.array(X_hog).shape}")

  N_COMPONENTS = 300
  svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
  X_hog_reduced = svd.fit_transform(X_hog)

  explained_variance_ratio = svd.explained_variance_ratio_.sum()
  print(f"Variance explained: {explained_variance_ratio:.4f} ({explained_variance_ratio*100:.2f}%)")
  print(f"Reduced HOG features: shape = {X_hog_reduced.shape}")

  hog_svd_model = {
    'svd': svd
  }
  with open('hog_svd_model.pkl', 'wb') as f:
    pickle.dump(hog_svd_model, f)