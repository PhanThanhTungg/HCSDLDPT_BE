import numpy as np
from pre_processing import bgr_to_gray

def manual_sobel_x(image):
	if len(image.shape) > 2:
			raise ValueError("Đầu vào phải là ảnh grayscale")
	image = image.astype(np.float64)
	height, width = image.shape
	gradient_x = np.zeros((height, width), dtype=np.float64)
	for y in range(height):
			for x in range(1, width - 1):
					gradient_x[y, x] = image[y, x + 1] - image[y, x - 1]
	for y in range(height):
			gradient_x[y, 0] = image[y, 1] - image[y, 0]
	for y in range(height):
			gradient_x[y, width - 1] = image[y, width - 1] - image[y, width - 2]
	return gradient_x

def manual_sobel_y(image):
	if len(image.shape) > 2:
			raise ValueError("Đầu vào phải là ảnh grayscale")
	image = image.astype(np.float64)
	height, width = image.shape
	gradient_y = np.zeros((height, width), dtype=np.float64)

	for y in range(1, height - 1):
			for x in range(width):
					gradient_y[y, x] = image[y + 1, x] - image[y - 1, x]

	for x in range(width):
			gradient_y[0, x] = image[1, x] - image[0, x]
	
	for x in range(width):
			gradient_y[height - 1, x] = image[height - 1, x] - image[height - 2, x]
	
	return gradient_y


def extract_hog_features(image, cell_size=8, block_size=2, bins=9):
	if len(image.shape) == 3:
		gray = bgr_to_gray(image)
	else:
		gray = image.copy()
		
	height, width = gray.shape
	
	gx = manual_sobel_x(gray)
	gy = manual_sobel_y(gray)

	magnitude = np.sqrt(gx**2 + gy**2)
	orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
	
	n_cells_y = height // cell_size
	n_cells_x = width // cell_size
	
	hog_cells = np.zeros((n_cells_y, n_cells_x, bins))
	
	for y in range(n_cells_y):
		for x in range(n_cells_x):
			cell_magnitude = magnitude[y*cell_size:(y+1)*cell_size, 
										x*cell_size:(x+1)*cell_size]
			cell_orientation = orientation[y*cell_size:(y+1)*cell_size, 
											x*cell_size:(x+1)*cell_size]
			
			for i in range(cell_size):
				for j in range(cell_size):
					if y*cell_size+i < height and x*cell_size+j < width:
						grad_mag = cell_magnitude[i, j]
						grad_ang = cell_orientation[i, j]
						
						bin_index = int(grad_ang / (180.0 / bins)) % bins
						hog_cells[y, x, bin_index] += grad_mag
	
	n_blocks_y = n_cells_y - block_size + 1
	n_blocks_x = n_cells_x - block_size + 1
	normalized_blocks = np.zeros((n_blocks_y, n_blocks_x, block_size * block_size * bins))
	
	for y in range(n_blocks_y):
		for x in range(n_blocks_x):
			block = hog_cells[y:y+block_size, x:x+block_size, :].flatten()
			block_norm = np.sqrt(np.sum(block**2) + 1e-10)
			if block_norm > 0:
				block = block / block_norm
			normalized_blocks[y, x, :] = block
	
	hog_features = normalized_blocks.flatten()
	
	return hog_features