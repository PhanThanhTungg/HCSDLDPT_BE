import numpy as np
import cv2

# extract uniform LBP features
def extract_uniform_lbp(image, radius=2, points=8):
	if len(image.shape) == 3:
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	else:
		gray = image
	
	height, width = gray.shape
	lbp_image = np.zeros((height, width), dtype=np.uint8)
	
	uniform_patterns = []
	for i in range(0, 2**points):
		bitstring = format(i, f'0{points}b')
		transitions = 0
		for j in range(points):
			transitions += (bitstring[j] != bitstring[(j+1) % points])
		
		if transitions <= 2:
			uniform_patterns.append(i)
	
	n_patterns = len(uniform_patterns)
	uniform_map = {pattern: idx for idx, pattern in enumerate(uniform_patterns)}
	
	for y in range(radius, height - radius):
		for x in range(radius, width - radius):
			center = gray[y, x]
			binary = 0
			
			for p in range(points):
				angle = 2 * np.pi * p / points
				x_p = int(round(x + radius * np.cos(angle)))
				y_p = int(round(y + radius * np.sin(angle)))
				
				if gray[y_p, x_p] >= center:
					binary |= (1 << p)
			
			if binary in uniform_map:
				lbp_image[y, x] = uniform_map[binary]
			else:
				lbp_image[y, x] = n_patterns
	
	hist, _ = np.histogram(lbp_image, bins=n_patterns+1, range=(0, n_patterns))
	
	hist = hist.astype("float") / hist.sum()
	
	return hist

