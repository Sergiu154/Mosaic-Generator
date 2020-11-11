# Mosaic Generator

Given an input image, a dataset of smaller images and the width of the final image, the algorithm found in **add_pieces_mosaic.py** builds a mosaic which is composed of smaller images and resembles the original input. Such a mosaic has multiple pattern options which are determined by the shape of the subimages(rectangular or hexagonal) and their positioning(with or without different neighbors). Tha final image is upscaled such that the width = n*piece_width and its aspect ratio is preserved. The path to the image collection, input image and other options are found in **run_project.py**.

###### To build your own mosaic type
```
python3 run_project.py
```


![Cat of cats]
(https://postimg.cc/gwkjcNKB)

![hexagon_pieces]
(https://postimg.cc/xX9TgtBT)

