# Virtual Draw

A webcam-based drawing canvas controlled with hand gestures. The project uses MediaPipe hand landmarks and OpenCV to turn finger movement into strokes, color selection, erasing, and canvas controls.

## Features

- Real-time hand tracking from a webcam
- Draw with an index-finger gesture
- Palette selection and eraser mode
- Two-finger color wheel with custom color selection
- Hold an open hand to clear the canvas
- Make a fist to pause drawing
- Stroke smoothing and basic line/circle correction
- On-screen gesture status, notifications, and clear progress
- Two-hand detection for deliberate canvas clearing

## Setup

Requirements: Python 3.10+ and a webcam.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

The latest prototype is:

```bash
python virtual_idea_presenter22.py
```

Press `q` to quit.

## Gesture guide

| Gesture | Action |
|---|---|
| Index finger up | Draw or select a palette item |
| Index and middle fingers up | Open and use the color wheel |
| Fist | Pause |
| Open hand held briefly | Clear the canvas |

For best results, use even lighting, keep the hand fully inside the frame, and place the camera near eye level.

## Project notes

This repository includes earlier experiments alongside the current prototype so the evolution of the interaction model remains visible. `virtual_idea_presenter22.py` is the recommended entry point.

## Tech stack

Python · OpenCV · MediaPipe · NumPy
