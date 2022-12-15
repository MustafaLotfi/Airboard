# Project: Write on Air
# Programmer: Mostafa Lotfi
# date: 12/15/2022
# Brief description: A little and cool project to write something on air and maybe erase it or change colors.
# opencv and mediapipe were used for working with frames and finding hand landmarks.
# Because of mediapipe light models, the project can work realtime.

from codes.write import Write


wrt = Write()

wrt.running()