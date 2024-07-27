from django.shortcuts import render,redirect
from django.shortcuts import render
from django.http import HttpResponse
# import tensorflow as tf
import numpy as np
import logging

import os

from django.core.files.uploadedfile import TemporaryUploadedFile
from django.core.files.uploadedfile import UploadedFile
from PIL import Image
from django.http import JsonResponse
# from tensorflow.keras.models import load_model
from tempfile import NamedTemporaryFile
from .models import userreg
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.core.files.storage import FileSystemStorage

import cv2
import numpy as np
import mediapipe as mp
import os

# Create your views here.
def home(request):
    return render(request,'home.html')


def register(request):
    if request.method=="POST":
       username=request.POST.get('username')
       email=request.POST.get('email')
       password=request.POST.get('password')
       userreg(username=username,email=email,password=password).save()
       return redirect('login')
    return render(request,'register.html')

def login(request):
    return render(request,'login.html')

def log(request):
    if request.method=="POST":
        username=request.POST.get('username')
        password=request.POST.get('password')
        stud=userreg.objects.filter(username=username,password=password)
        if stud:
            return redirect('upload')
        else:
            return render(request,'login.html')
    else:
        return render(request,'register.html')
    



# Define the joint list and other constants
joint_list = [[8, 7, 6, 5], [12, 11, 10, 9], [16, 15, 14, 13], [20, 19, 18, 17]]

# Function to draw finger angles on the image
def draw_finger_angles(image, results, joint_list):
    mp_hands = mp.solutions.hands

    for hand in results.multi_hand_landmarks:
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])  # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])  # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])  # Third coord
            d = np.array([hand.landmark[joint[3]].x, hand.landmark[joint[3]].y])  # Fourth coord

            radians_1 = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle_1 = np.abs(radians_1 * 180.0 / np.pi)
            radians_2 = np.arctan2(d[1] - c[1], d[0] - c[0]) - np.arctan2(b[1] - c[1], b[0] - c[0])
            angle_2 = np.abs(radians_2 * 180.0 / np.pi)

            if angle_1 > 180.0:
                angle_1 = 360 - angle_1

            if angle_2 > 180.0:
                angle_2 = 360 - angle_2

            if angle_1 < 175:
                cv2.putText(image, str(round(angle_1, 2)),
                            tuple(np.multiply(b, [image.shape[1], image.shape[0]]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(image, str(round(angle_1, 2)),
                            tuple(np.multiply(b, [image.shape[1], image.shape[0]]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            if angle_2 < 175:
                cv2.putText(image, str(round(angle_2, 2)),
                            tuple(np.multiply(c, [image.shape[1], image.shape[0]]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(image, str(round(angle_2, 2)),
                            tuple(np.multiply(c, [image.shape[1], image.shape[0]]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return image

# Main view function


logger = logging.getLogger(__name__)

def blood_cell_detection(request):
    if request.method == 'POST' and request.FILES['image']:
        try:
            logger.info('Image upload detected.')
            mp_drawing = mp.solutions.drawing_utils
            mp_hands = mp.solutions.hands

            mp_model = mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5
            )

            image_file = request.FILES['image']
            logger.info('Image file received: %s', image_file.name)

            image_data = image_file.read()
            nparr = np.fromstring(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = cv2.flip(image, 1)
            logger.info('Image decoded and flipped.')

            results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        image,
                        hand,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                    )
                draw_finger_angles(image, results, joint_list)
                logger.info('Hand landmarks drawn and angles calculated.')

            fs = FileSystemStorage()
            output_path = os.path.join(settings.MEDIA_ROOT, 'processed_image.jpg')
            cv2.imwrite(output_path, image)
            logger.info('Processed image saved at %s', output_path)

            context = {'output_path': fs.url('processed_image.jpg')}
            return render(request, 'result.html', context)

        except Exception as e:
            logger.error('Error processing image: %s', str(e))
            return HttpResponse('Error processing image', status=500)

    return render(request, 'index.html')


def processed_image(request):
    image_path = os.path.join(settings.MEDIA_ROOT, 'image.jpg')
    with open(image_path, "rb") as f:
        return HttpResponse(f.read(), content_type="image/jpeg")





