import cv2
import os


def capture_images_auto(person_name, num_images=20):
    person_folder = f"dataset/{person_name}"
    os.makedirs(person_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    image_counter = 0

    while image_counter < num_images:
        ret, frame = cap.read()

        cv2.imshow('Capture Images - Automatic', frame)

        image_filename = f"{person_folder}/img_{image_counter}.jpg"
        cv2.imwrite(image_filename, frame)
        print(f"Image captured: {image_filename}")

        image_counter += 1
        cv2.waitKey(100)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    person_name = input("Enter the person's name: ")
    capture_images_auto(person_name, num_images=20)
