# Mosaic Generator

Given an input image, a dataset of smaller images and the width of the final image, the algorithm found in **add_pieces_mosaic.py** builds a mosaic which is composed of smaller images and resembles the original input. Such a mosaic has multiple pattern options which are determined by the shape of the subimages(rectangular or hexagonal) and their positioning(with or without different neighbors). Tha final image is upscaled such that the width = n*piece_width and its aspect ratio is preserved. The path to the image collection, input image and other options are found in **run_project.py**.

###### To build your own mosaic type
```
python3 run_project.py
```


![caroiaj_dist_standing_john_100](https://user-images.githubusercontent.com/21235087/98833121-56f2de80-2446-11eb-8435-c61a347a7482.png)


![hexagon_dist_tom_100](https://user-images.githubusercontent.com/21235087/98833173-640fcd80-2446-11eb-9f96-4bec8a2bd0bc.png)

