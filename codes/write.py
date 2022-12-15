import mediapipe as mp
import cv2
import numpy as np
import datetime
import os


INDEX_FINGER_ID = 8
THUMB_ID = 4
FINGERS_MIN_DIST = 0.05
CIRCLE_SIZE = 5
FONT_SIZE = 1
SAVING_FOLDER = "files"
PEN_REG = ((0.0, 0.15), (0.1, 0.2))
ERASER_REG = ((0.0, 0.15), (0.3, 0.4))
COLOR1_REG = ((0.0, 0.15), (0.5, 0.6))
COLOR2_REG = ((0.0, 0.15), (0.7, 0.8))

PEN_CLR = (0, 0, 0)
ERASER_CLR = (0, 0, 0)
COLOR1_CLR = (255, 255, 0)
COLOR2_CLR = (0, 255, 255)

PEN_TXT = "PEN"
ERASER_TXT = "ERS"
COLOR1_TXT = "CLR1"
COLOR2_TXT = "CLR2"

regions = (PEN_REG, ERASER_REG, COLOR1_REG, COLOR2_REG)
colors = (PEN_CLR, ERASER_CLR, COLOR1_CLR, COLOR2_CLR)
texts = (PEN_TXT, ERASER_TXT, COLOR1_TXT, COLOR2_TXT)

ERASE_AREA = (0.05, 0.05)

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

		self.color = COLOR1_CLR
		self.pen_or_erase = 0

		self.cap = cv2.VideoCapture(camera_id)

		self.app_running = True

		self.points_list = []

		self.bolds = [1, 0, 1, 0]
			
		create_dir(SAVING_FOLDER)
		new_file_number = get_new_file_number(SAVING_FOLDER)

		self.video = cv2.VideoWriter(SAVING_FOLDER + "/" + new_file_number + ".avi",
			cv2.VideoWriter_fourcc(*"MJPG"), 14, (640, 480))


	def running(self):
		while self.cap.isOpened() and self.app_running:
			ret, self.frame = self.cap.read()

			if ret:
				self.get_frames()

				self.frame = self.add_default_shapes(self.frame)

				self.get_landmarks()

				self.calc_fingers_features()

				self.check_modes()

				self.add_remove_landmarks()

				self.add_landmarks2img()
				
				self.show_save_frames()

		self.finalizing()


	def get_frames(self):
		self.frame_size = self.frame.shape[1], self.frame.shape[0]
		self.frame = cv2.flip(self.frame, 1)
		self.frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)


	def add_default_shapes(self, frame):
		frs = frame.shape[1], frame.shape[0]
		i = 0
		for (reg, clr, txt) in zip(regions, colors, texts):
			thickness = 2
			if self.bolds[i] == 1:
				thickness = 4

			p1 = int(reg[0][0] * frs[0]), int(reg[1][0] * frs[1])
			p2 = int(reg[0][1] * frs[0]), int(reg[1][1] * frs[1])
			cv2.rectangle(frame, p1, p2, clr, thickness)

			p = int((reg[0][0] + 0.02) * frs[0]), int((reg[1][0] + 0.07) * frs[1])
			cv2.putText(frame, txt, p, cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, clr, 2)

			i += 1

		return frame


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


	def calc_fingers_features(self):
		if self.found:
			index_finger = self.landmarks[INDEX_FINGER_ID, :2]
			thumb = self.landmarks[THUMB_ID, :2]

			self.fingers_middle = (index_finger + thumb) / 2

			self.fingers_dist = self.calc_dist(index_finger, thumb)

			self.are_close = self.fingers_dist < FINGERS_MIN_DIST

		
	def add_remove_landmarks(self):
		if self.found and self.are_close and self.write_screen:
			if self.pen_or_erase == 0:
				self.points_list.append(self.fingers_middle)
			else:
				points = np.array(self.points_list)
				del_points_ids = []
				for (i, pnt) in enumerate(points):
					cns1 = (self.fingers_middle[0] - ERASE_AREA[0]/2) < pnt[0] < (self.fingers_middle[0] + ERASE_AREA[0]/2)
					cns2 = (self.fingers_middle[1] - ERASE_AREA[1]/2) < pnt[1] < (self.fingers_middle[1] + ERASE_AREA[1]/2)
					if cns1 and cns2:
						del_points_ids.append(i)

				points = np.delete(points, del_points_ids, 0)

				self.points_list = list(points)


	def check_modes(self):
		if self.found:
			self.write_screen = True

			cns1 = regions[0][0][0] < self.fingers_middle[0] < regions[0][0][1]
			cns2 = regions[0][1][0] < self.fingers_middle[1] < regions[0][1][1]
			if self.are_close and cns1 and cns2:
				self.bolds[0] = 1
				self.bolds[1] = 0
				self.write_screen = False
				self.pen_or_erase = 0

			cns1 = regions[1][0][0] < self.fingers_middle[0] < regions[1][0][1]
			cns2 = regions[1][1][0] < self.fingers_middle[1] < regions[1][1][1]
			if self.are_close and cns1 and cns2:
				self.bolds[0] = 0
				self.bolds[1] = 1
				self.write_screen = False
				self.pen_or_erase = 1

			cns1 = regions[2][0][0] < self.fingers_middle[0] < regions[2][0][1]
			cns2 = regions[2][1][0] < self.fingers_middle[1] < regions[2][1][1]
			if self.are_close and cns1 and cns2:
				self.bolds[0] = 1
				self.bolds[1] = 0
				self.bolds[2] = 1
				self.bolds[3] = 0
				self.write_screen = False
				self.color = COLOR1_CLR
				self.pen_or_erase = 0

			cns1 = regions[3][0][0] < self.fingers_middle[0] < regions[3][0][1]
			cns2 = regions[3][1][0] < self.fingers_middle[1] < regions[3][1][1]
			if self.are_close and cns1 and cns2:
				self.bolds[0] = 1
				self.bolds[1] = 0
				self.bolds[2] = 0
				self.bolds[3] = 1
				self.write_screen = False
				self.color = COLOR2_CLR
				self.pen_or_erase = 0


	def add_landmarks2img(self):
		for point in self.points_list:
			fr_point = (point * self.frame_size).astype(np.uint32)
			cv2.circle(self.frame, fr_point, CIRCLE_SIZE, self.color, cv2.FILLED)


	@staticmethod
	def calc_dist(p1, p2):
		return np.sqrt(((p1-p2)**2).sum())


	def finalizing(self):
		self.cap.release()
		cv2.destroyAllWindows()
		self.video.release()

