import os
import face_recognition
from sklearn import neighbors
import pickle


def train_model():
    X = []
    y = []

    for person in os.listdir("dataset"):
        for file in os.listdir(f"dataset/{person}"):
            image_path = f"dataset/{person}/{file}"
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)

            if len(encoding) > 0:
                X.append(encoding[0])
                y.append(person)

    model = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree')
    model.fit(X, y)

    # Save the model
    with open('face_recognition_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    with open('known_encodings.pkl', 'wb') as encodings_file:
        pickle.dump(X, encodings_file)

    with open('known_names.pkl', 'wb') as names_file:
        pickle.dump(y, names_file)


if __name__ == "__main__":
    train_model()
