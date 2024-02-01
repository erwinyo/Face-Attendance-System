from flask import Flask, render_template, Response
import cv2

from recognition import FaceRecognition

app = Flask(__name__)
fr = FaceRecognition()


@app.route('/')
def index():
    return "This page show that the server is running :)"


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_frames():
    for frame in fr.run_recognition():
        # Convert the OpenCV image (BGR format) to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


if __name__ == '__main__':
    app.run(port=8000, debug=True)
