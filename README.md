# Write-On-Air
## Introduction
A little and cool project to write something on air and maybe erase it or change colors.
opencv and mediapipe were used for working with frames and finding hand landmarks. Because of
mediapipe light models, the project can work realtime.

## Preview

## How to run
Clone the repository:

`git clone <repo url>`

Create and activate a virtual environment

On windows:

`python -m venv <venv name>`

`.\venv\Scripts\activate`

Install required packages:

`pip install -r requirements.txt`

Make sure your webcam is setting up, then run main.py:

`python main.py`
