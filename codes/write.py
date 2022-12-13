import mediapipe as mp
import cv2
import numpy as np
import datetime
import os


INDEX_FINGER_ID = 8
THUMB_ID = 4
FINGERS_MIN_DIST = 0.05
SAVING_FOLDER = "media"


def create_dir(name):
	exist = False
	current_dir = os.path.dirname(__file__) + "/../"
	files_name = os.listdir(current_dir)
	for fn in files_name:
		if fn == name:
			exist = True
			break

	if not exist:
		os.mkdir(SAVING_FOLDER)


def get_new_file_number(name):
	fol_dir = os.path.dirname(__file__) + "/../" + name
	files_name = os.listdir(fol_dir)

	files_numbers = []
	if files_name:
		for fn in files_name:
			files_numbers.append(int(fn[:-4]))

		new_file_number = str(max(files_numbers)+1)
	else:
		new_file_number = "0"

	return new_file_number


class Write():
	def __init__(self, camera_id=0):
		self.hands = mp.solutions.hands.Hands(min_detection_confidence=0.5)

		self.cap = cv2.VideoCapture(camera_id)

		self.app_running = True

		self.points_list = []
			
		create_dir(SAVING_FOLDER)
		new_file_number = get_new_file_number(SAVING_FOLDER)

		self.video = cv2.VideoWriter(SAVING_FOLDER + "/" + new_file_number + ".avi",
			cv2.VideoWriter_fourcc(*"MJPG"), 14, (640, 480))


	def running(self):
		while self.cap.isOpened() and self.app_running:
			ret, self.frame = self.cap.read()

			if ret:
				self.get_frames()

				self.get_landmarks()

				self.add_landmarks2list()

				self.add_landmarks2img()
				
				self.show_save_frames()

		self.finalizing()



	def get_frames(self):
		self.frame_size = self.frame.shape[1], self.frame.shape[0]
		self.frame = cv2.flip(self.frame, 1)
		self.frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)


	def show_save_frames(self):
		cv2.imshow("Camera", self.frame)
		q = cv2.waitKey(1)

		if (q == ord('q')) or (q == ord('Q')):
			self.app_running = False

		self.video.write(self.frame)


	def get_landmarks(self):
		self.found = False

		lms = self.hands.process(self.frame_rgb).multi_hand_landmarks

		if lms:
			self.found = True

			self.landmarks = np.array(
				[[point.x, point.y, point.z] for point in lms[0].landmark])

		
	def add_landmarks2list(self):
			if self.found:
				index_finger = self.landmarks[INDEX_FINGER_ID, :2]
				thumb = self.landmarks[THUMB_ID, :2]

				fingers_middle = (index_finger + thumb) / 2

				fingers_dist = self.calc_dist(index_finger, thumb)

				if fingers_dist < FINGERS_MIN_DIST:
					self.points_list.append(fingers_middle)


	def add_landmarks2img(self):
		for point in self.points_list:
			fr_point = (point * self.frame_size).astype(np.uint32)
			cv2.circle(self.frame, fr_point, 5, (255, 255, 0), cv2.FILLED)


	@staticmethod
	def calc_dist(p1, p2):
		return np.sqrt(((p1-p2)**2).sum())


	def finalizing(self):
		self.cap.release()
		cv2.destroyAllWindows()
		self.video.release()





