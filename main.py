import numpy as np

class Settings:
	content_image = 'input/content_img.jpg'
	style_image = 'input/style_img.jpg'
	stop_stage = 0.0005
	max_num_stage = 1000
	output = 'output/out.jpg'
	
	def get_settings():
		return np.array(content_image, style_image, stop_stage, max_num_stage, output)

