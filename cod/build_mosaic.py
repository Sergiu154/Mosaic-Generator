import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pdb

from cod.add_pieces_mosaic import *
from cod.parameters import *
from glob import glob


def load_pieces(params: Parameters):
    # citeste toate cele N piese folosite la mozaic din directorul corespunzator
    # toate cele N imagini au aceeasi dimensiune H x W x C, unde:
    # H = inaltime, W = latime, C = nr canale (C=1  gri, C=3 color)
    # functia intoarce pieseMozaic = matrice N x H x W x C in params
    # pieseMoziac[i, :, :, :] reprezinta piesa numarul i

    images = np.array([np.stack([cv.imread(img_path, cv.IMREAD_GRAYSCALE) for _ in range(3)],
                                axis=-1) if params.grayscale_flag else cv.imread(img_path)
                       for img_path in glob(params.small_images_dir + '*.png')])

    # citeste imaginile din director

    # if params.show_small_images:
    #     for i in range(10):
    #         for j in range(10):
    #             plt.subplot(10, 10, i * 10 + j + 1)
    #             # OpenCV reads images in BGR format, matplotlib reads images in RBG format
    #             im = images[i * 10 + j].copy()
    #             # BGR to RGB, swap the channels
    #             im = im[:, :, [2, 1, 0]]
    #             plt.imshow(im)
    #     plt.show()

    params.small_images = images


def compute_dimensions(params: Parameters):
    # calculeaza dimensiunile mozaicului
    # obtine si imaginea de referinta redimensionata avand aceleasi dimensiuni
    # ca mozaicul

    # completati codul
    # calculeaza automat numarul de piese pe verticala
    h, w = params.image.shape[:2]
    small_img_h, small_img_w = params.small_images[0].shape[:2]
    ratio = w / h

    # redimensioneaza imaginea
    new_w = small_img_w * params.num_pieces_horizontal
    new_h = int(new_w // ratio)

    params.num_pieces_vertical = int(new_h // small_img_h)

    params.image_resized = cv.resize(params.image, (new_w, new_h))


def build_mosaic(params: Parameters):
    # incarcam imaginile din care vom forma mozaicul
    load_pieces(params)
    # calculeaza dimensiunea mozaicului
    compute_dimensions(params)

    img_mosaic = None
    if params.layout == 'caroiaj':
        if params.hexagon is True:
            img_mosaic = add_pieces_hexagon(params)
        else:
            img_mosaic = add_pieces_grid(params)
    elif params.layout == 'aleator':
        img_mosaic = add_pieces_random(params)
    else:
        print('Wrong option!')
        exit(-1)

    return img_mosaic
