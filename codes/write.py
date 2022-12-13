import mediapipe as mp
import cv2
import numpy as np


class Write():
	def __init__(self, camera_id=0):
		self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.5)

		self.cap = cv2.VideoCapture(camera_id)

		self.app_is_running = True


	def running(self):
		while cap.isOpened() and self.app_is_running:
			ret, frame = self.cap.read()

			if ret:
				frame = cv2.flip(frame, 0)

				cv2.imshow(frame)
				q = cv2.waitKey(1)

				if (q == ord('q')) or (q == ord('Q')):
					self.app_is_running = False