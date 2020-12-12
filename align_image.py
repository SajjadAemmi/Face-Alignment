import os
import argparse

from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector


def main(RAW_IMAGES_DIR,ALIGNED_IMAGES_DIR,landmarks_model_path):
    """
        Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
        python align_images.py /raw_images /aligned_images
        """

    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for img_name in os.listdir(RAW_IMAGES_DIR):
        raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
            face_img_name = ('%s.'+img_name.split('.')[-1]) % (os.path.splitext(img_name)[0])
            aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
            image_align(raw_img_path, aligned_face_path, face_landmarks)
            break


if __name__ == "__main__":

    landmarks_model_path = 'models/shape_predictor_68_face_landmarks.dat'

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input image path', default='input/source.jpg', type=str)
    parser.add_argument('--output', help='output image path', default='output/source.jpg', type=str)
    args = parser.parse_args()

    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(args.input), start=1):
        image_align(args.input, args.output, face_landmarks)
