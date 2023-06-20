from flask import Flask, request
import numpy as np
import cv2
import easyocr
import imutils
import tempfile
import os
import eventlet
from eventlet import wsgi
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
reader = easyocr.Reader(['en'])


def data_preprocessing(image):
  location = None
  img = cv2.imread(image)
  greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  bfilter = cv2.bilateralFilter(greyscale_img, 11, 17, 17)
  edged = cv2.Canny(bfilter, 30, 30)
  keypoints = cv2.findContours(
  edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours = imutils.grab_contours(keypoints)
  contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

  for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4: 
      location = approx
      break

  mask = np.zeros(greyscale_img.shape, np.uint8)
  new_image = cv2.drawContours(mask, [location], 0, 255, -1,)
  new_image = cv2.bitwise_and(img, img, mask=mask)
  (x, y) = np.where(mask == 255)
  (x1, y1) = (np.min(x), np.min(y))
  (x2, y2) = (np.max(x), np.max(y))
  cropped_image = greyscale_img[x1:x2+1, y1:y2+1]

  return cropped_image


def detect(image):
  image = data_preprocessing(image)
  result = reader.readtext(image)
  return result


@app.route('/detect', methods=["POST"])
def index():
  try:
    image = request.files.get('image')
    # Create a temporary file
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, image.name)

    # Save the file temporarily
    image.save(temp_file_path)
    result = detect(temp_file_path)
    os.remove(temp_file_path)

    if len(result) > 0:
      return {'message': f'The licence plate of the car is: {result[0][1]}'}
      
    return {'message': 'could not detect licence plate'}
  except Exception as e:
    return {'error': f'failed to process image {e}'}


if __name__ == '__main__':
  wsgi.server(eventlet.listen(('', 5000)), app)
