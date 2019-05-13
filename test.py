import argparse
from src.rectifyAffine.rectifyAffine import RectifyAffine

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='src/images/images_bordas/')
    parser.add_argument('--image_name', type=str, default='DSC_000016608')
    args = parser.parse_args()

    rectAffine = RectifyAffine(path=args.path, image_name=args.image_name)
    img_result = rectAffine.image_rectification()
    rectAffine.estimated_speed(img_result)
    print("Finished...!")

