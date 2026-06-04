import cv2
from load_image import load_raw
from optimisation import optimise_parameters
from processing import process_image

PATH = "test_images/DSCF4785.RAF"

def main():
    image = load_raw(PATH)
    
    smaller_img = cv2.resize(image, (1508, 1007))
    best_params = optimise_parameters(smaller_img, n_trials=100)
    
    print("Best parameters: ", best_params)
    
    best_image = process_image(image, best_params)
    
    cv2.imwrite("result.jpeg", best_image)
    print("Saved result.jpeg")
    

if __name__ == "__main__":
        main()