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

def histogram_equalization_color(img):
  out = np.zeros_like(img)
  for c in range(3):
      out[..., c] = histogram_equalization(img[..., c])
  return out