import pprint
import matplotlib.pyplot as plt
import requests
from matplotlib import patches
from PIL import Image

ENDPOINT = "https://cviprojectfaceapi.cognitiveservices.azure.com/"
SUBSCRIPTION_KEY = "d97ba7e737ee492db8e2fab623f3e86a"
FACE_API_URL = ENDPOINT + "/face/v1.0/detect"

# Load Image
image_filepath = './Images/No Mask/e3a7151e-4f2c-11ec-ad44-70665522a7c4.jpg'
pil_image = Image.open(image_filepath, 'r')

# Show Image
plt.figure(figsize=(5, 5))
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
    plt.figure(figsize=(5, 5))
    plt.imshow(pil_image)

    for detected_face in response_data:
        rectangle_data =  detected_face['faceRectangle']
        x = rectangle_data['left']
        y = rectangle_data['top']
        w = rectangle_data['width']
        h = rectangle_data['height']
        if detected_face['faceAttributes']['mask']['type'] == 'noMask':
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


plot_image_with_mask_label(r'C:\Users\Ayushi\OneDrive\Desktop\CVI\Project\Images\Test\a0eaab7b-5326-11ec-b1d4-70665522a7c4.jpg')
plot_image_with_mask_label(r'C:\Users\Ayushi\OneDrive\Desktop\CVI\Project\Images\Black Mask\9f0f24bc-4f2c-11ec-bd69-70665522a7c4.jpg')
plt.show()
