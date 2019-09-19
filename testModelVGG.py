# python testModelVGG.py --model vggModel.model --image images/h.jpg

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from keras.engine.training import Model

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
orig = image.copy()

image = cv2.resize(image, (64, 64))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print("[INFO] loading network...")
model = load_model(args["model"])

(volkswagen, hyundai, porsche, honda, audi) = model.predict(image)[0]
print("=================================================")
print("Prediction: ")
print("=================================================")
print("Honda: ", honda)
print("Volkswagen: ", volkswagen)
print("Hyundai: ", hyundai)
print("Porsche: ", porsche)
print("Audi: ", audi)

# cv2.imshow("Image", image)
# cv2.waitKey(0)