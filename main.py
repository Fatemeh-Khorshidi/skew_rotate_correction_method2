import argparse
import imutils
import cv2
from PIL import Image
import imutils
import os
import time
from google.colab.patches import cv2_imshow
from skewcurrection.skew_stimation import estimate_skewness, de_skew
from orientationdetection.orientationdetection import OrientationDetection

def load_image(path):
    try:
        img = cv2.imread(path)
        # perform BRG to gray scale conversion,we need only gray scale information
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # invert the color; so that the backgorund becomes dark and foreground text is white
        Inverted_Gray = cv2.bitwise_not(gray)

        # covert the gray scale image to binary image; here, 0 is the min-thresold for binarization (adjustable, but usually small)
        # returns the binarized image vector to "binary"
        binary = cv2.threshold(Inverted_Gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # dilation = cv2.dilate(binary,(5,5),iterations = 2) # dilate the white pixels of image
        # binary = dilation
        return img, binary
    except:
        print("Could not load the image; provide correct path")
        return None, None



def main(img_dir, new_dir):
    os.makedirs(new_dir, exist_ok=True)
    count = 0
    start = time.time()
    for image in os.listdir(img_dir):
        if image.endswith(("jpg", "png")):
            image_dir = os.path.join(img_dir, image)
            print(image_dir)
            img, binary = load_image(image_dir)
            if img is None or binary is None:
                return None

            angle = estimate_skewness(binary)
            rotated_image = de_skew(img, angle)
            detector = OrientationDetection(rotated_image)
            bbox = detector.text_detection()

            # Rotate the image if it's vertical
            detector.left_right_rotation()

            # Rotate the image if it's upside down
            rotated_image = detector.up_down_rotate()
            count += 1

             # Save the image with the original file name
            image_name = os.path.splitext(image)[0]
            new_image_path = os.path.join(new_dir, f'{image_name}.jpg')
            rotated_image = Image.fromarray(rotated_image)
            rotated_image.save(new_image_path)

    end = time.time()
    total_time = end - start
    print("total time", total_time)


def parse_args():
    parser = argparse.ArgumentParser(description="rotate and skew currection")
    parser.add_argument("--image_dir", required=True, help="Path to the input directory scaned document image")
    parser.add_argument("--save_dir", required=True, help="Path to the output directory scaned document image")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    model_path = '/content/drive/MyDrive/SystemGroupPrj-20231028T181858Z-001/SystemGroupPrj/rotate_model.h5'
    args = parse_args()
    main(args.image_dir, args.save_dir)
