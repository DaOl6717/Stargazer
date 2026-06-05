import cv2
import numpy as np

def image_score(image):
    image = neutralise_colour(image) # Prova ta bort?

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    stars = count_stars(image)

    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    noise = np.mean(np.abs(gray - blur))

    brightness = gray.mean()

    score = (
        5.0 * stars
        - 5.0 * noise
        - 0.1 * abs(brightness - 65)
    )

    return score

def count_stars(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stars = 0

    for c in contours:
        area = cv2.contourArea(c)

        if 3 < area < 40:
            stars += 1

    return stars

def est_noise(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (9, 9), 0)
    return np.mean(np.abs(grey - blur))

def colour_balance_penalty(image):
    b, g, r = cv2.split(image)
    
    mean_r = r.mean()
    mean_g = g.mean()
    mean_b = b.mean()
     
    penalty = abs(mean_r - mean_g) + abs(mean_r - mean_b) + abs(mean_g - mean_b)
    
    return penalty

def neutralise_colour(image):
    img = image.astype("float32")
    b, g, r = cv2.split(img)
    
    mean = (b.mean() + g.mean() + r.mean()) / 3.0
    
    b *= mean / (b.mean() + 1e-6)
    g *= mean / (g.mean() + 1e-6)
    r *= mean / (r.mean() + 1e-6)

    img = cv2.merge([b, g, r])
    return np.clip(img, 0, 255).astype(np.uint8)
