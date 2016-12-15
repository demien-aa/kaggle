import json
import cv2
from matplotlib import pyplot as plt


with open('data/lag_labels.json') as json_data:
    lag_json = json.load(json_data)
print(lag_json)
# file_path = 'data/train/LAG/img_00091.jpg'
# img = cv2.imread(file_path, 0)
# cv2.rectangle(img,(537, 214), (948, 323), 255, 2)
# plt.imshow(img, cmap='gray')
# plt.show()
