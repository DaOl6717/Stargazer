import cv2
import numpy as np

def stretch(img, strength):
    return np.log1p(strength * img) / np.log1p(strength)

def post_stretch_denoise(img):
    return cv2.fastNlMeansDenoisingColored(
        (img * 255).astype("uint8"),
        None,
        h=5,
        hColor=5,
        templateWindowSize=7,
        searchWindowSize=21
        ).astype(np.float32) / 255.0


def process_image(img, params):
    img = img.copy()
    
    # Normalise
    img = img / 255.0
    
    # Black level
    img = remove_black_level(img)
    
    # White balance
    img = auto_white_balance(img)
    
    # Denoise
    img = selective_denoise(img, 0.6)
    
    # Exposure
    img = img * params["exposure"]
    
    # Tone mapping
    img = filmic_tone_map(img)
    
    # Chroma denoise
    img = remove_chroma_noise(img)
    
    img = denoise_luminance(img)
    
    # Fix sky
    img = sky_colour_normalisation(img)

    # Denoise
    img = selective_denoise(img, 0.3)

    # Sharpen
    kernel = np.array([
        [0, -1, 0],
        [-1, 4 + params["sharpen"], -1],
        [0, -1, 0]
        ])
    img = cv2.filter2D(img, -1, kernel)
    
    img = compress_midtones(img)

    return (np.clip(img, 0, 1) * 255).astype("uint8")

def apply_gamma(img, gamma):
    return np.power(img, gamma)

def color_balance(img, r_gain, g_gain, b_gain):
    img[:, :, 0] *= b_gain
    img[:, :, 1] *= g_gain
    img[:, :, 2] *= r_gain
    
    return np.clip(img, 0, 1)

def get_background_mask(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(grey, 100, 255, cv2.THRESH_BINARY_INV)
    
    return mask

def remove_chroma_noise(img, strength=1.0):
    lab = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Strong smoothing of ONLY color channels
    a = cv2.GaussianBlur(a, (0,0), 3 * strength)
    b = cv2.GaussianBlur(b, (0,0), 3 * strength)

    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return result.astype("float32") / 255.0

def selective_denoise(img, strength):
    img_uint8 = (img * 255).astype("uint8")
    
    mask = get_background_mask(img_uint8)
    
    denoised = cv2.fastNlMeansDenoisingColored(img_uint8, None, h=10 * strength,
                                               hColor=10 * strength, templateWindowSize=7,
                                               searchWindowSize=21)
    
    result = np.where(mask[:,:,None] == 255, denoised, img_uint8)
    
    return result.astype("float32") / 255.0

def auto_white_balance(img):
    img = img.astype("float32")
    b, g, r = cv2.split(img)
    
    mean = (b.mean() + g.mean() + r.mean()) / 3.0
    
    b *= mean / (b.mean() + 1e-6)
    g *= mean / (g.mean() + 1e-6)
    r *= mean / (r.mean() + 1e-6)

    img = cv2.merge([b, g, r])
    return np.clip(img, 0, 1)

def filmic_tone_map(x, exposure=1.0):
    x = x * exposure
    return x / (1.0 + x)

def remove_black_level(img):
    black = np.percentile(img, 10)
    img = img - black
    return np.clip(img, 0, 1)

def sky_colour_normalisation(img):
    b, g, r = cv2.split(img)

    r *= 0.9
    g *= 0.95
    b *= 1.05

    img = cv2.merge([b, g, r])

    return np.clip(img, 0, 1)


def denoise_luminance(img):
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l = cv2.GaussianBlur(l, (0,0), 1.0)

    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return result.astype(np.float32) / 255.0

def compress_midtones(img):
    return np.power(img, 1.2)
