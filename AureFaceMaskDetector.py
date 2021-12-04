import pprint
import matplotlib.pyplot as plt
import requests
from matplotlib import patches
from PIL import Image
import os
import random
import cv2
import uuid
import time
import glob

ENDPOINT = "https://faceapicvi.cognitiveservices.azure.com/"
SUBSCRIPTION_KEY = "12d992d085a64df08abb2b5da5dc45cf"
FACE_API_URL = ENDPOINT + "/face/v1.0/detect"

files = glob.glob('./Images/Test/*.jpg', recursive=True)
for f in files:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))

# Load Image
image_filepath = './Images/No Mask/e3a7151e-4f2c-11ec-ad44-70665522a7c4.jpg'
pil_image = Image.open(image_filepath, 'r')

# Show Image
plt.figure(figsize=(10, 5))
plt.imshow(pil_image)

headers = {
    'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY,
    'Content-Type': 'application/octet-stream'
}

params = {
    'returnFaceId': 'false',
    'returnFaceLandmarks': 'true',
    'returnFaceAttributes': 'age,gender,glasses,smile'
}

image_data = open(image_filepath, 'rb').read()

response = requests.post(
    FACE_API_URL,
    params=params,
    headers=headers,
    data=image_data
)

response_data = response.json()
pprint.pprint(response_data)

for detected_face in response_data:
    rect = patches.Rectangle(
        [
            detected_face['faceRectangle']['left'],
            detected_face['faceRectangle']['top']
        ],
        detected_face['faceRectangle']['width'],
        detected_face['faceRectangle']['height'],
        linewidth=2, edgecolor='blue', facecolor='none'
    )
    ax = plt.gca()
    ax.add_patch(rect)


def call_face_api(image_filepath):
    params = {
        'returnFaceId': 'false',
        'returnFaceLandmarks': 'true',
        'returnFaceAttributes': 'mask',
        'detectionModel': 'detection_03'
    }

    image_data = open(image_filepath, 'rb').read()

    response = requests.post(
        FACE_API_URL,
        params=params,
        headers=headers,
        data=image_data
    )
    response_data = response.json()
    return response_data


def plot_image_with_mask_label(image_filepath):
    response_data = call_face_api(image_filepath)
    pil_image = Image.open(image_filepath, 'r')
    plt.figure(figsize=(15, 7))
    plt.imshow(pil_image)

    for detected_face in response_data:
        rectangle_data =  detected_face['faceRectangle']
        x = rectangle_data['left']
        y = rectangle_data['top']
        w = rectangle_data['width']
        h = rectangle_data['height']
        if detected_face['faceAttributes']['mask']['type'] == 'noMask' or detected_face['faceAttributes']['mask']['noseAndMouthCovered'] == 0:
            label_str = "No Mask"
            color_str = "red"
        else:
            label_str = "Wearing Mask"
            color_str = "limegreen"
        rect = patches.Rectangle(
            [x, y], w, h,
            linewidth=2, edgecolor=color_str, facecolor='none'
        )
        ax = plt.gca()
        plt.text(
            x,
            y + h + 60,
            label_str,
            size=15,
            c=color_str
        )
        ax.add_patch(rect)

# Define the duration (in seconds) of the video capture here
capture_duration = 5
cap = cv2.VideoCapture(0)
start_time = time.time()

if(cap.isOpened()==False):
    print("Error opening video stream")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while(cap.isOpened()):
    while( int(time.time() - start_time) < capture_duration ):
        #Captureframe-by-frame
        ret, frame = cap.read()
        img_name = './Images/Test/{}.jpg'.format(str(uuid.uuid1()))
        cv2.imwrite(img_name, frame)
        cv2.imshow('frame', frame)

    break

cap.release()
cv2.destroyAllWindows()

i=0
path = r"C:\Users\Ayushi\OneDrive\Desktop\CVI\Project\Images\Test"

while i<3:
    random_filename = random.choice([
        x for x in os.listdir(path)
        if os.path.isfile(os.path.join(path, x))
    ])
    image_path = str(path + '\\' + random_filename)
    plot_image_with_mask_label(image_path)
    plt.show()
    i += 1

