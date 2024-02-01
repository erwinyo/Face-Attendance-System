import threading

import cv2
from flask import Flask, redirect
from flask import Response

from recognition import FaceRecognition

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
fr = FaceRecognition()


@app.route("/")
def index():
    return "This is testing page. It tell you this is working :)"


@app.route("/raw")
def raw():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start")
def start():
    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, lock

    # Start a thread that will perform motion detection
    t = threading.Thread(target=fr.run_recognition)
    t.daemon = True
    t.start()

    for frame in fr.run_recognition():
        with lock:
            outputFrame = frame.copy()

    return redirect("/raw")


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    while True:
        # check if the output frame is available, otherwise skip
        # the iteration of the loop
        if outputFrame is None:
            continue

        # encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        # ensure the frame was successfully encoded
        if not flag:
            continue

        # yield the output frame in the byte format
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'


# check to see if this is the main thread of execution
if __name__ == '__main__':
    """
        host : 0.0.0.0 
        - this is a must, cannot be changed to 127.0.0.1 
        - or it will cannot be accessed after been forwarded by docker to host IP

        port : 80 (up to you)
    """
    app.run(host="0.0.0.0", port=80, debug=True, threaded=True, use_reloader=False)
