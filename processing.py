import cv2
import numpy as np

def process_image(image, params):
    img = image.copy()

    # Normalise
    img = img / 255.0

    # Stretch
    img = np.log1p(params["stretch"] * img)
    img = img / img.max()

    # Denoise
    img = selective_denoise(img, params["denoise"])

    # Contrast
    img = np.clip((img - 0.5) * params["contrast"] + 0.5, 0, 1)

    # Sharpen
    kernel = np.array([
        [0, -1, 0],
        [-1, 5 + params["sharpen"], -1],
        [0, -1, 0]
        ])
    img = cv2.filter2D(img, -1, kernel)

    return (img * 255).astype(np.uint8)

def apply_gamma(image, gamma):
    return np.power(image, gamma)

def color_balance(img, r_gain, g_gain, b_gain):
    img[:, :, 0] *= b_gain
    img[:, :, 1] *= g_gain
    img[:, :, 2] *= r_gain
    
    return np.clip(img, 0, 1)

def get_background_mask(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(grey, 100, 255, cv2.THRESH_BINARY_INV)
    
    return mask

def selective_denoise(img, strength):
    img_uint8 = (img * 255).astype("uint8")
    
    mask = get_background_mask(img_uint8)
    
    denoised = cv2.fastNlMeansDenoisingColored(img_uint8, None, h=10 * strength,
                                               hColor=10 * strength, templateWindowSize=7,
                                               searchWindowSize=21)
    
    result = np.where(mask[:,:,None] == 255, denoised, img_uint8)
    
    return result.astype("float32") / 255.0