import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def contourDetection(img):
    
    # Conversion to grayscale 
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy() 

    # Blurr
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    # Find Canny edges 
    edged = cv2.Canny(blur, 30, 200) 
    cv2.imshow('Canny', edged) 

    # Finding Contours 
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    #cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    #cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    print("Number of Contours found = " + str(len(contours))) 
    
    # Create a copy of the image to draw contours on
    img_with_contours = img.copy()
    
    # Draw all contours (i.e -1)
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 3) 
    
    cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)  
    cv2.imshow('Contours', img_with_contours)
    
    return img_with_contours, edged

def main(args):
    
    ## Open and resize image
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'cameraman.tif')
    
    base_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Check if image was loaded successfully
    if base_img is None:
        print(f"Erreur: Impossible de charger l'image '{image_path}'")
        print("Vérifiez que le fichier cameraman.tif existe dans le répertoire tp3/")
        return 1
    
    # Handle both grayscale and color images
    if len(base_img.shape) == 3:
        h, w, c = base_img.shape
    else:
        h, w = base_img.shape
        c = 1
    
    width = 512 # specify new width
    scale_f = width/w
    height = h*scale_f
    base_img = cv2.resize(base_img, (int(width),int(height)))
    
    # Convert grayscale to BGR if needed (for colored contour drawing)
    if len(base_img.shape) == 2:
        img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    else:
        img = base_img.copy()

    cv2.imshow('orig', img)
    
    
    ## Process image
    img_with_contours, edges_only = contourDetection(img)
    
    # Save the image with contours drawn on original
    output_path = os.path.join(script_dir, 'cameraman_contours.png')
    cv2.imwrite(output_path, img_with_contours)
    print(f"Image avec contours sauvegardée: {output_path}")
    
    # Save the image with only the contours (edges)
    edges_output_path = os.path.join(script_dir, 'cameraman_edges_only.png')
    cv2.imwrite(edges_output_path, edges_only)
    print(f"Image avec uniquement les contours sauvegardée: {edges_output_path}")
    
    ## wait ESC or q
    key = cv2.waitKey(0) & 0x0FF # wait indefinitely and get input
    if key == 27 or key == ord('q'): #key is ESC or q
        cv2.destroyAllWindows()
    
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))