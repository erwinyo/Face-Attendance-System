import os
import sys

import cv2
import face_recognition
import numpy as np

from recognition import util


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_face()

    def encode_face(self):
        for image in os.listdir("asset/faces/"):
            face_image = face_recognition.load_image_file(f"asset/faces/{image}")
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit("Could not open video")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            if self.process_current_frame:
                # Resize image for save resources
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # Change BGR to RGB
                small_frame = small_frame[:, :, ::-1]

                # Find all faces in current frame
                self.face_locations = face_recognition.face_locations(small_frame)
                self.face_encodings = face_recognition.face_encodings(small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = "Unknown"

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    # If matched
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = util.face_confidence(face_distances[best_match_index])

                    self.face_names.append(f"{name} : {confidence}")

            # Skip one frame
            # Reduce the resources consumption
            self.process_current_frame = not self.process_current_frame

            # Annotation
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Time 4, because previous code we resize to the 0.25x0.25
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

            yield frame

        video_capture.release()
        cv2.destroyAllWindows()
