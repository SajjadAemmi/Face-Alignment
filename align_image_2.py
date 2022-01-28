import os
import argparse

from imutils.face_utils import FaceAligner
import imutils
import dlib
import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to input image")
    parser.add_argument('--output', help='output image path', default='output/', type=str)
    parser.add_argument("-p", "--shape-predictor", default="models/shape_predictor_68_face_landmarks.dat", help="path to facial landmark predictor")
    args = parser.parse_args()

    file_name, file_ext = os.path.splitext(os.path.basename(args.input))

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    image = cv2.imread(args.input)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 2)

    for i, rect in enumerate(rects):
        faceAligned = fa.align(image, gray, rect)
        
        output_file_path = os.path.join(args.output, file_name + "_" + str(i) + ".jpg")
        cv2.imwrite(output_file_path, faceAligned)
