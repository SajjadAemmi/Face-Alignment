import os
import argparse
import cv2
from tqdm import tqdm
from face_alignment import align_face
from landmarks_detector import LandmarksDetector


def align_image(input_image_path, output_dir_path):
    try:
        file_name, file_ext = os.path.splitext(os.path.basename(input_image_path))
        all_face_landmarks = landmarks_detector.get_landmarks(input_image_path)
        for i, face_landmarks in enumerate(all_face_landmarks):
            image = align_face(input_image_path, face_landmarks)
            output_file_path = os.path.join(output_dir_path, file_name + "_" + str(i) + ".jpg")
            cv2.imwrite(output_file_path, image)
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Alignment")
    parser.add_argument('--input', help='input image or directory path', default='./input/scarlett-johansson.jpg', type=str)
    parser.add_argument('--output', help='output image path', default='./output', type=str)
    parser.add_argument('--landmarks-model-path', help='landmarks model path', default='models/shape_predictor_68_face_landmarks.dat', type=str)
    args = parser.parse_args()

    landmarks_detector = LandmarksDetector(args.landmarks_model_path)

    if os.path.isfile(args.input):  # single image
        align_image(args.input, args.output)

    elif os.path.isdir(args.input):  # images dir
        for path, directories, files in os.walk(args.input):
            for directory in directories:
                destination = os.path.join(args.output, directory)
                os.makedirs(destination, exist_ok=True)
            print(path)      
            for file in tqdm(files):
                input_file_path = os.path.join(path, file)
                output_dir_path = os.path.join(args.output, os.path.relpath(path, args.input))
                align_image(input_file_path, output_dir_path)
