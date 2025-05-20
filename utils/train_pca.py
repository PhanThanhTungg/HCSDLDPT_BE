import os
import cv2
import numpy as np

def load_images(folder, size):
  images = []
  for filename in os.listdir(folder):
      path = os.path.join(folder, filename)
      img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
      if img is not None:
          img = cv2.resize(img, size)
          images.append(img.flatten())
  return np.array(images)

def train_pca(image_size=(64, 64), n_components=150):
  IMAGE_SIZE = image_size
  N_COMPONENTS = n_components
  X = load_images('Data', IMAGE_SIZE)
  print(f"Loaded {X.shape[0]} images, mỗi ảnh có {X.shape[1]} pixel")

  mean_vector = np.mean(X, axis=0)
  X_centered = X - mean_vector

  cov_matrix = np.cov(X_centered, rowvar=False)  

  eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

  idx_sorted = np.argsort(eigenvalues)[::-1]
  eigenvectors_sorted = eigenvectors[:, idx_sorted]
  eigenvalues_sorted = eigenvalues[idx_sorted]

  components = eigenvectors_sorted[:, :N_COMPONENTS]

  X_pca = np.dot(X_centered, components)

  total_variance = np.sum(eigenvalues)
  retained_variance = np.sum(eigenvalues_sorted[:N_COMPONENTS])
  retained_ratio = retained_variance / total_variance

  print(f"Tổng phương sai giữ lại: {retained_ratio:.4f} ({retained_ratio*100:.2f}%)")

  print(f"Dữ liệu sau PCA: shape = {X_pca.shape}")