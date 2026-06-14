# Advanced Air Drawing

An AI-powered virtual whiteboard that lets you draw in the air with a webcam and hand gestures.

## Features

- Draw with your index finger
- Select from multiple colors using pinch gestures
- Adjust brush thickness with the on-screen slider or `+` / `-`
- Use an eraser gesture with index + middle finger
- Clear the canvas with an open palm or the `C` key
- Real-time hand tracking with MediaPipe

## Controls

- Index finger only: draw
- Index + middle fingers: eraser
- Open palm: clear canvas
- Pinch over a color swatch: select that color
- Pinch and drag the vertical slider: change brush thickness
- `C`: clear canvas
- `Q` or `Esc`: quit

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the app:

   ```bash
   python main.py
   ```

## Notes

- Use a well-lit background for best tracking.
- Keep one hand in frame for the most stable results.
