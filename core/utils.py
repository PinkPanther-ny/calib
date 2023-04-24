import cv2


def combine_image(image1, image2, alpha1=0.5, alpha2=0.5, gamma=0):
    # gamma value (scalar added to each sum)
    return cv2.addWeighted(image1, alpha1, image2, alpha2, gamma)
