import os
import argparse

from face_alignment import image_align
from landmarks_detector import LandmarksDetector
import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input image path', default='input/source.jpg', type=str)
    parser.add_argument('--output', help='output image path', default='output/', type=str)
    parser.add_argument('--landmarks-model-path', help='landmarks model path', default='models/shape_predictor_68_face_landmarks.dat', type=str)
    args = parser.parse_args()

    file_name, file_ext = os.path.splitext(os.path.basename(args.input))
    landmarks_detector = LandmarksDetector(args.landmarks_model_path)

    try:
        all_face_landmarks = landmarks_detector.get_landmarks(args.input)
        for i, face_landmarks in enumerate(all_face_landmarks):
            image = image_align(args.input, face_landmarks)

            output_file_path = os.path.join(args.output, file_name + "_" + str(i) + ".jpg")
            cv2.imwrite(output_file_path, image)

    except Exception as e:
        print("Error:", e)
