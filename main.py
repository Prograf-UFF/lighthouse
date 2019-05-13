from src.rectifyAffine.rectifyAffine import RectifyAffine

if __name__ == "__main__":
    path = 'src/images/images_bordas/'
    img = 'DSC_000016608'

    rectAffine = RectifyAffine(path=path, image_name=img)
    img_result = rectAffine.image_rectification()
    rectAffine.estimated_speed(img_result)
    print("Finished...!")

