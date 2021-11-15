#! /usr/env/python
"""Take all images in the directory and align them."""
import os
import argparse
import pathlib

from face_alignment import image_align
from landmarks_detector import LandmarksDetector
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', help='input image path', default='input/', type=str)
    parser.add_argument('--ext', help='input file extension', default='jpg', type=str)
    parser.add_argument('--output', help='output image path', default='output/', type=str)
    parser.add_argument('--landmarks-model-path', help='landmarks model path', default='models/shape_predictor_68_face_landmarks.dat', type=str)
    args = parser.parse_args()

    landmarks_detector = LandmarksDetector(args.landmarks_model_path)

    path = pathlib.Path(args.input_dir)
    file_ext = args.ext
    jpg_files = path.glob(f"*.{file_ext}")
    for picture in jpg_files:
        file_name = picture.stem
        print(file_name, picture)
        try:
            all_face_landmarks = landmarks_detector.get_landmarks(str(picture))
            for i, face_landmarks in enumerate(all_face_landmarks):
                image = image_align(str(picture), face_landmarks)
                output_file_path = os.path.join(args.output, f"{file_name}-{i}.jpg")
                cv2.imwrite(output_file_path, image)

        except Exception as e:
            print("Error:", e)
