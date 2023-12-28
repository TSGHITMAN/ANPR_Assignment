import cv2
import face_recognition
import pickle
import numpy as np


def recognize_faces(frame, model, known_encodings, known_names):
    locations = face_recognition.face_locations(frame)
    encodings = face_recognition.face_encodings(frame, locations)

    names = []
    for face_encoding in encodings:
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        min_distance_index = int(np.argmin(face_distances))
        name = known_names[min_distance_index] if face_distances[min_distance_index] < 0.6 else "Unknown"
        names.append(name)

    return names


def main():
    with open('face_recognition_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('known_encodings.pkl', 'rb') as encodings_file:
        known_encodings = pickle.load(encodings_file)

    with open('known_names.pkl', 'rb') as names_file:
        known_names = pickle.load(names_file)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error reading frame from camera")
            break

        names = recognize_faces(frame, model, known_encodings, known_names)

        for (top, right, bottom, left), name in zip(face_recognition.face_locations(frame), names):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            name_str = str(name)

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name_str, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv2.imshow('Face Recognition - Press q to exit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
