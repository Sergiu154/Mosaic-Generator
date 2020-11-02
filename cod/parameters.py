import cv2 as cv
import numpy as np


# In aceasta clasa vom stoca detalii legate de algoritm si de imaginea pe care este aplicat.
class Parameters:

    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.image = cv.imread(image_path)

        if self.image is None:
            print('%s is not valid' % image_path)
            exit(-1)

        # verifica daca imaginea este grayscale si seteaza flagul corespunzator
        # daca imaginea e grayscale trebuie sa facem si img. din colectie la fel
        B, G, R = self.image[:, :, 0], self.image[:, :, 1], self.image[:, :, 2]
        self.grayscale_flag = ((B == G).all() and (G == R).all())

        self.image_resized = None
        self.small_images_dir = './../data/colectie/'
        self.image_type = 'png'
        self.num_pieces_horizontal = 100
        self.num_pieces_vertical = None
        self.show_small_images = False
        self.layout = 'caroiaj'
        self.criterion = 'aleator'
        self.hexagon = False
        self.small_images = None
        self.dist_neighbor = False
