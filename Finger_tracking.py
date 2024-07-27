#!/usr/bin/env python
# coding: utf-8




# !pip install mediapipe
from matplotlib import pyplot as plt
from matplotlib import image
import numpy as np
    
import cv2
import mediapipe as mp



joint_list = [[8,7,6,5], [12,11,10,9], [16,15,14,13], [20,19,18,17]]




def draw_finger_angles(request, image, results, joint_list):
    
    
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            d = np.array([hand.landmark[joint[3]].x, hand.landmark[joint[3]].y]) #fourth coord

            radians_1 = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle_1 = np.abs(radians_1*180.0/np.pi)
            radians_2 = np.arctan2(d[1] - c[1], d[0]-c[0]) - np.arctan2(b[1]-c[1], b[0]-c[0])
            angle_2 = np.abs(radians_2*180.0/np.pi)

            if angle_1 > 180.0:
                angle_1 = 360-angle_1
                
                    
            if angle_2 > 180.0:
                angle_2 = 360-angle_2
            
            if angle_1<175:
                    cv2.putText(image, str(round(angle_1, 2)), tuple(np.multiply(b, [image.shape[1], image.shape[0]]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(image, str(round(angle_1, 2)), tuple(np.multiply(b, [image.shape[1], image.shape[0]]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            if angle_2<175:
                    cv2.putText(image, str(round(angle_2, 2)), tuple(np.multiply(c, [image.shape[1], image.shape[0]]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(image, str(round(angle_2, 2)), tuple(np.multiply(c, [image.shape[1], image.shape[0]]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    
            
           
            
    return image




mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
mp_model = mp_hands.Hands(
    static_image_mode=True, # only static images
    max_num_hands=2, # max 2 hands detection
    min_detection_confidence=0.5) # detection confidence

image_file = request.FILES['image']

image = cv2.imread(image_file.path)

image = cv2.flip(image, 1)

results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))



# Detections
# print(results)
# print(results.multi_handedness)




image_height, image_width, _ = image.shape
annotated_image = image.copy()
for hand_landmarks in results.multi_hand_landmarks:
    print('hand_landmarks:', hand_landmarks)
    print(
      f'Index finger tip coordinates: (',
      f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
      f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
    )
    

if results.multi_hand_landmarks:
    for num, hand in enumerate(results.multi_hand_landmarks):
        mp_drawing.draw_landmarks(
            image, 
            hand, 
            mp_hands.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
        )

    
    # Draw angles to image from joint list
    draw_finger_angles(image, results, joint_list)



cv2.imshow('Hand Tracking', image)
cv2.waitKey(0)













