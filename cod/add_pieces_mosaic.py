from cod.parameters import *
import numpy as np
import pdb
import timeit

"""
    cropped_image: a patch from the original image
    remainder: if the last small image does not fit entirely then remainder !=0 and return a cropped small image
    first_nth: used when we want to have diff. neighbors, by default return the first most similar image  
    
"""


def get_most_similar(cropped_image, image_collection, remainder=0, first_nth=1):
    temp = np.array(sorted(image_collection,
                           key=lambda x:
                           np.sqrt(np.sum(
                               cv.absdiff(np.array(cv.mean(x)[:-1]), np.array(cv.mean(cropped_image)[:-1])) ** 2))))
    # if there the last small image does not fit then return a cropped small image
    if remainder:
        return temp[:first_nth, :remainder, :]
    return temp[:first_nth, :, :]


def preprocess_hexagonal_mosaic(params):
    small_image_h, small_image_w = params.small_images[0].shape[:2]
    new_h, new_w = params.image_resized.shape[:2]

    # daca imaginea e patrata o redimensionam dreptunghiulara
    # pt. a putea extrage un romb "perfect" pt imagini generale
    if small_image_h == small_image_w:
        for i in range(len(params.small_images)):
            params.small_images[i] = cv.resize(params.small_images[i], (small_image_h, 1.25 * small_image_w))

        small_image_h, small_image_w, _ = params.small_images.shape

    # atlfel daca imaginea e dreptunghiulara putem extrage rombul
    lateral_mid = small_image_h // 2
    mask = create_hexagon_mask(small_image_h, small_image_w)

    # mareste imagine originala pentru a putea incapea si coltuirle hex ca ies din imagine
    # la sfarsit taiem acele parti pentru a avea din nou dimensiunile originale

    hex_H, hex_W = new_h + 2 * lateral_mid, new_w + 2 * lateral_mid

    print('Dimensiunea viitorului mozic hex este ', hex_H, hex_W)

    new_hex_image = np.zeros(shape=(hex_H, hex_W, 3), dtype='uint8')
    new_hex_image[lateral_mid:hex_H - lateral_mid, lateral_mid:hex_W - lateral_mid, :] = params.image_resized.copy()

    # copieza pe noul border dark format colonele/liniile de la marginea imaginii
    for i in range(lateral_mid, hex_H - lateral_mid):
        new_hex_image[i, 0: lateral_mid, :] = new_hex_image[i, lateral_mid:2 * lateral_mid, :]
        new_hex_image[i, hex_W - lateral_mid: hex_W, :] = new_hex_image[i, hex_W - 2 * lateral_mid: hex_W - lateral_mid,
                                                          :]
    for j in range(0, hex_W):
        new_hex_image[0:lateral_mid, j, :] = new_hex_image[lateral_mid:2 * lateral_mid, j, :]
        new_hex_image[hex_H - lateral_mid:hex_H, j, :] = new_hex_image[hex_H - 2 * lateral_mid:hex_H - lateral_mid, j,
                                                         :]
    hexagon = np.zeros(shape=(hex_H, hex_W, 3), dtype='uint8')
    return hex_H, hex_W, hexagon, lateral_mid, mask, new_hex_image, small_image_h, small_image_w


def find_first_diff_neighbor(similar_images, neighbors):
    for img in similar_images:
        ok = True
        for neighbor in neighbors:
            # daca este egal cu unul din vecini atunci nu e bun
            if neighbor is not None:
                if np.array_equal(img, neighbor):
                    ok = False
                    break
        # daca am gasit o poza care e diferita de vecini o returnam
        if ok:
            return img.copy()


def create_hexagon_mask(small_image_h, small_image_w):
    mask = np.empty(shape=(small_image_h, small_image_w, 3), dtype='uint8')
    mask.fill(255)

    lateral_mid = small_image_h // 2

    upper_left = mask[:lateral_mid, :lateral_mid, :]
    lower_right = mask[lateral_mid:, small_image_w - lateral_mid:small_image_w, :]
    upper_right = mask[:lateral_mid, small_image_w - lateral_mid:small_image_w, :]
    lower_left = mask[lateral_mid:, :lateral_mid, :]

    for i in range(3):
        t = np.empty(shape=(lateral_mid, lateral_mid))
        t.fill(255)
        upper_right[:, :, i] = np.tril(t)

    for i in range(3):
        t = np.empty(shape=(lateral_mid, lateral_mid))
        t.fill(255)
        upper_left[:, :, i] = np.fliplr(np.tril(t))

        # daca luam diagonala si in upper si in lower atunci se suprapun
        # pentru lowert luam fara diagonala
    for i in range(3):
        t = np.empty(shape=(lateral_mid, lateral_mid))
        t.fill(255)
        lower_right[:, :, i] = np.flipud(np.tril(t, k=-1))

    for i in range(3):
        t = np.empty(shape=(lateral_mid, lateral_mid))
        t.fill(255)
        lower_left[:, :, i] = np.flipud(np.fliplr(np.tril(t, k=-1)))

    mask[mask == 255] = 1
    return mask


def grid_image(params, new_h, new_w, small_img_h, small_img_w):
    caroiaj = np.zeros(shape=(new_h, new_w, 3), dtype='uint8')

    for i in range(params.num_pieces_vertical + 1):
        for j in range(params.num_pieces_horizontal):
            h_piece = min(new_h - i * small_img_h, small_img_h)
            if h_piece == 0:
                return caroiaj

            print("H piece", h_piece)
            cropped_image = params.image_resized[i * small_img_h:i * small_img_h + h_piece,
                            j * small_img_w:(j + 1) * small_img_w, :]
            chosen_image = get_most_similar(cropped_image, params.small_images,
                                            remainder=h_piece if small_img_h - h_piece != 0 else 0)[0]

            caroiaj[i * small_img_h:i * small_img_h + h_piece,
            j * small_img_w:(j + 1) * small_img_w, :] = chosen_image.copy()

    # remainder = new_h % small_img_h
    #
    # if remainder:
    #     for j in range(params.num_pieces_horizontal):
    #         cropped_image = params.image_resized[
    #                         params.num_pieces_vertical * small_img_h:params.num_pieces_vertical * small_img_h + remainder,
    #                         j * small_img_w:(j + 1) * small_img_w, :]
    #         chosen_image = get_most_similar(cropped_image, params.small_images, remainder)[0]
    #
    #         caroiaj[params.num_pieces_vertical * small_img_h:params.num_pieces_vertical * small_img_h + remainder,
    #         j * small_img_w:(j + 1) * small_img_w, :] = chosen_image.copy()

    # cv.imwrite('../Caroiaj_ferrari_100.png', caroiaj)

    cv.imshow('Caroiaj', caroiaj)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return caroiaj


def grid_image_with_diff_neighbors(params, new_h, new_w, small_img_h, small_img_w):
    caroiaj_distinct = np.zeros(shape=(new_h, new_w, 3), dtype='uint8')

    remainder = new_h % small_img_h

    # chenar cu vecini diferiti
    for i in range(params.num_pieces_vertical + 1):
        for j in range(params.num_pieces_horizontal):

            h_piece = min(new_h - i * small_img_h, small_img_h)
            if h_piece == 0:
                return caroiaj_distinct

            cropped_image = params.image_resized[i * small_img_h: i * small_img_h + h_piece,
                            j * small_img_w:(j + 1) * small_img_w, :]
            similar_images = get_most_similar(cropped_image, params.small_images,
                                              remainder=h_piece if small_img_h - h_piece != 0 else 0,
                                              first_nth=len(params.image_resized))

            left_neighbour, upper_neighbour = None, None
            if i == 0 and j == 0:
                # daca e piesa din coltul stanga sum
                print(similar_images.shape)
                caroiaj_distinct[i * small_img_h: i * small_img_h + h_piece, j * small_img_w:(j + 1) * small_img_w, :] = \
                    similar_images[0].copy()

            # daca e piesa de pe prima linie verific doar vecin stang
            else:
                if i == 0:
                    left_neighbour = caroiaj_distinct[i * small_img_h:i * small_img_h + h_piece,
                                     (j - 1) * small_img_w:j * small_img_w, :]

                    chosen_image = find_first_diff_neighbor(similar_images, [left_neighbour])

                elif j == 0:
                    upper_neighbour = caroiaj_distinct[(i - 1) * small_img_h:(i - 1) * small_img_h + h_piece,
                                      j * small_img_w:(j + 1) * small_img_w, :]

                    chosen_image = find_first_diff_neighbor(similar_images, [upper_neighbour])

                else:
                    left_neighbour = caroiaj_distinct[i * small_img_h:i * small_img_h + h_piece,
                                     (j - 1) * small_img_w:j * small_img_w, :]
                    upper_neighbour = caroiaj_distinct[(i - 1) * small_img_h:(i - 1) * small_img_h + h_piece,
                                      j * small_img_w:(j + 1) * small_img_w, :]

                    chosen_image = find_first_diff_neighbor(similar_images, [left_neighbour, upper_neighbour])

                caroiaj_distinct[i * small_img_h:i * small_img_h + h_piece, j * small_img_w:(j + 1) * small_img_w,
                :] = chosen_image.copy()

    # if remainder:
    #     for j in range(params.num_pieces_horizontal):
    #         cropped_image = params.image_resized[
    #                         params.num_pieces_vertical * small_img_h:params.num_pieces_vertical * small_img_h + remainder,
    #                         j * small_img_w:(j + 1) * small_img_w, :]
    #
    #         similar_images = get_most_similar(cropped_image, params.small_images, remainder=remainder,
    #                                           first_nth=len(params.image_resized))
    #
    #         if j == 0:
    #             upper_neighbour = caroiaj_distinct[(params.num_pieces_vertical - 1) * small_img_h:
    #                                                params.num_pieces_vertical * small_img_h,
    #                               j * small_img_w:(j + 1) * small_img_w, :]
    #             for img in similar_images:
    #                 if not np.array_equal(img, upper_neighbour):
    #                     chosen_image = img.copy()
    #                     break
    #
    #         else:
    #             left_neighbour = caroiaj_distinct[small_img_h * small_img_h:(small_img_h + 1) * small_img_h,
    #                              (j - 1) * small_img_w:j * small_img_w, :]
    #             upper_neighbour = caroiaj_distinct[(small_img_h - 1) * small_img_h:small_img_h * small_img_h,
    #                               j * small_img_w:(j + 1) * small_img_w, :]
    #
    #             for img in similar_images:
    #                 if not np.array_equal(img, left_neighbour) and not np.array_equal(img, upper_neighbour):
    #                     chosen_image = img.copy()
    #                     break
    #
    #         caroiaj_distinct[
    #         params.num_pieces_vertical * small_img_h:params.num_pieces_vertical * small_img_h + remainder,
    #         j * small_img_w:(j + 1) * small_img_w, :] = chosen_image.copy()

    cv.imshow('Caroiaj', caroiaj_distinct)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # cv.imwrite('../Distinct_fe_100.png', caroiaj_distinct)

    return caroiaj_distinct


def add_pieces_grid(params: Parameters):
    start_time = timeit.default_timer()
    img_mosaic = np.zeros(params.image_resized.shape, np.uint8)
    new_h, new_w = params.image_resized.shape[:2]
    small_img_h, small_img_w = params.small_images[0].shape[:2]
    num_pieces = params.num_pieces_vertical * params.num_pieces_horizontal

    if params.criterion == 'aleator':
        for i in range(params.num_pieces_vertical):
            for j in range(params.num_pieces_horizontal):
                index = np.random.randint(low=0, high=params.small_images.shape[0], size=1)
                img_mosaic[i * new_h: (i + 1) * new_h, j * new_w: (j + 1) * new_w, :] = params.small_images[index]
                print('Building mosaic %.2f%%' % (100 * (i * params.num_pieces_horizontal + j + 1) / num_pieces))

    elif params.criterion == 'distantaCuloareMedie':

        if params.dist_neighbor:
            img_mosaic = grid_image_with_diff_neighbors(params, new_h, new_w, small_img_h, small_img_w)
        else:
            img_mosaic = grid_image(params, new_h, new_w, small_img_h, small_img_w)

    else:
        print('Error! unknown option %s' % params.criterion)
        exit(-1)

    end_time = timeit.default_timer()
    print('Running time: %f s.' % (end_time - start_time))

    return img_mosaic


def add_pieces_random(params: Parameters):
    # aleator

    new_h, new_w = params.image_resized.shape[:2]
    small_img_h, small_img_w = params.small_images[0].shape[:2]

    random_image = np.zeros(shape=(new_h, new_w, 3), dtype='uint8')

    mask = {}
    for i in range(new_h):
        mask[i] = {}
        for j in range(new_w):
            mask[i][j] = False

    while len(mask) > 0:
        print(len(mask))

        current_i = np.array([k for k, v in mask.items()])
        rand_i = np.random.choice(current_i)

        current_j = np.array([k for k, v in mask[rand_i].items()])
        rand_j = np.random.choice(current_j)

        step_x = min(small_img_h, new_h - rand_i)
        step_y = min(small_img_w, new_w - rand_j)

        cropped_image = params.image_resized[rand_i:rand_i + step_x, rand_j:rand_j + step_y, :].copy()

        chosen_image = get_most_similar(cropped_image, params.small_images)[0][:step_x, :step_y, :]

        random_image[rand_i:rand_i + step_x, rand_j:rand_j + step_y, :] = chosen_image.copy()
        for i in range(rand_i, rand_i + step_x):
            for j in range(rand_j, rand_j + step_y):
                if i in mask:
                    if j in mask[i]:
                        del mask[i][j]

            if i in mask:
                #                 print(len(mask[i]))
                if len(mask[i]) == 0:
                    del mask[i]
    cv.imwrite('../Random_Obama_100.png', random_image)
    # print(random_image)
    cv.imshow('asda', random_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def add_pieces_hexagon(params: Parameters):
    hex_H, hex_W, hexagon, lateral_mid, mask, new_hex_image, small_image_h, small_image_w = preprocess_hexagonal_mosaic(
        params)

    # latimea unei imagini este formata din 2*lateral_mid + restu
    # pasul pe latimea imaginii va fi deci de latime - lateral_mid = restul + lateral_mid
    step_j = small_image_w - lateral_mid
    end_W = hex_W // step_j

    # completam imaginea pe linii
    # alternan dintre linia "dreapta"
    # si cea cu hexagoanele in dreapta sus
    # variabile side ne spune pe ce tip de linie suntem
    # pe verticala step-ul este dat de inaltimea unei imagini mici

    # neighbors = np.zeros(shape=((i - lateral_mid) // small_image_h), )

    if not params.dist_neighbor:

        for i in range(lateral_mid, hex_H, small_image_h):
            side = 0
            for j in range(0, end_W * step_j + 1, step_j):
                #         rand_index = np.random.randint(0,len(image_collection))
                #         ran_img = image_collection[rand_index]

                step_x, step_y = min(hex_H - i, small_image_h), min(hex_W - j, small_image_w)
                #         print(step_x,step_y)
                cropped_image = new_hex_image[i - lateral_mid * side: i + step_x - lateral_mid * side, j:j + step_y,
                                :].copy()

                chosen_image = get_most_similar(cropped_image, params.small_images)[0]

                if side == 0:
                    #             print(hexagon[ i : i + step_x ,j : j+ step_y,:].shape)
                    #             print(np.multiply(final,chosen_image)[:step_x,:step_y,:].shape)
                    hexagon[i: i + step_x, j: j + step_y, :] += np.multiply(mask, chosen_image)[:step_x, :step_y, :]
                else:
                    #             up = 0 if i - offset == 0 else 1
                    hexagon[i - lateral_mid: i + step_x - lateral_mid, j:j + step_y, :] += np.multiply(mask,
                                                                                                       chosen_image)[
                                                                                           :step_x,
                                                                                           :step_y, :]

                side = 1 - side
    else:
        print("Vreau distinct")
        rest_width = small_image_w - lateral_mid

        for i in range(lateral_mid, hex_H, small_image_h):
            side = 0
            print(i)
            # if i == lateral_mid + small_image_h:
            #     cv.imwrite('../Hex_obama_100.png', hexagon)
            #     cv.imshow('asda', hexagon)
            #     cv.waitKey(0)

            for j in range(0, end_W * step_j + 1, step_j):
                #         rand_index = np.random.randint(0,len(image_collection))
                #         ran_img = image_collection[rand_index]

                step_x, step_y = min(hex_H - i, small_image_h), min(hex_W - j, small_image_w)
                #         print(step_x,step_y)
                cropped_image = new_hex_image[i - lateral_mid * side: i + step_x - lateral_mid * side, j:j + step_y,
                                :].copy()

                similar_images = get_most_similar(cropped_image, params.small_images,
                                                  remainder=0,
                                                  first_nth=len(params.image_resized))
                masked_similar_img = np.multiply(mask, similar_images)[:, :step_x, :step_y, :]

                # partea de jos cand incep
                if side == 0:
                    # cv.imshow('asda', hexagon)
                    # cv.waitKey(0)
                    # cv.destroyAllWindows()

                    if i == lateral_mid:
                        if j == 0:
                            # prima piesa din coltul stanga o punem ca atare
                            chosen_image = np.multiply(mask, similar_images[0].copy())

                        else:

                            # suntem pe prima linie pt side 0 , uita-te doar in stanga sus
                            print("STEP Y", step_y)

                            top_left_neigh = np.multiply(mask[:step_x, :step_y],
                                                         hexagon[i - lateral_mid: i - lateral_mid + step_x,
                                                         j - rest_width: j - rest_width + step_y])

                            chosen_image = find_first_diff_neighbor(
                                masked_similar_img, [top_left_neigh])
                    else:
                        # daca nu mai suntem pe prima linie
                        if j == 0:
                            # dar suntem pe prima coloana ne uitam doar in sus
                            upper_neigh = np.multiply(mask[:step_x, :step_y],
                                                      hexagon[i - small_image_h: i - small_image_h + step_x,
                                                      j: j + step_y])

                            chosen_image = find_first_diff_neighbor(masked_similar_img, [upper_neigh])

                        else:
                            # atunci verificam toti 2/3 vecini vecinii
                            down_left_neigh = None
                            # daca suntem pe ultima linie nu avem vecin in stanga jos
                            if i + lateral_mid + step_x <= hex_H:
                                down_left_neigh = np.multiply(mask[:step_x, :step_y],
                                                              hexagon[i + lateral_mid: i + lateral_mid + step_x,
                                                              j - rest_width:j - rest_width + step_y])
                            # rest_width = small_image_w - lateral_mid
                            top_left_neigh = np.multiply(mask[:step_x, :step_y],
                                                         hexagon[i - lateral_mid: i - lateral_mid + step_x,
                                                         j - rest_width: j - rest_width + step_y])
                            upper_neigh = np.multiply(mask[:step_x, :step_y],
                                                      hexagon[i - small_image_h: i - small_image_h + step_x,
                                                      j: j + step_y])

                            chosen_image = find_first_diff_neighbor(masked_similar_img,
                                                                    [down_left_neigh, top_left_neigh, upper_neigh])

                    #             print(hexagon[ i : i + step_x ,j : j+ step_y,:].shape)
                    #             print(np.multiply(final,chosen_image)[:step_x,:step_y,:].shape)
                    hexagon[i: i + step_x, j: j + step_y, :] += np.multiply(mask[:step_x, :step_y, :], chosen_image)[
                                                                :step_x, :step_y, :]
                # partea de sus cand incep
                else:
                    # verific cei 3/4 vecini

                    if i - lateral_mid == 0:
                        # daca sunt pe prima linie de la side 1 atunci ma uit doar in stanga jos
                        down_left_neigh = np.multiply(mask[:step_x, :step_y],
                                                      hexagon[i: i + step_x, j - rest_width:j - rest_width + step_y])

                        chosen_image = find_first_diff_neighbor(masked_similar_img, [down_left_neigh])

                    else:
                        # daca sunt la ultima coloana atunci nu am vecin dreapta sus
                        upper_right_neigh = None
                        if j + lateral_mid + step_y <= hex_W:
                            upper_right_neigh = np.multiply(mask[:step_x, :step_y],
                                                            hexagon[i - small_image_h:i - small_image_h + step_x,
                                                            j + rest_width: j + rest_width + step_y])

                        down_left_neigh = np.multiply(mask[:step_x, :step_y],
                                                      hexagon[i: i + step_x, j - rest_width:j - rest_width + step_y])
                        # rest_width = small_image_w - lateral_mid
                        top_left_neigh = np.multiply(mask[:step_x, :step_y],
                                                     hexagon[i - small_image_h: i - small_image_h + step_x,
                                                     j - rest_width: j - rest_width + step_y])
                        upper_neigh = np.multiply(mask[:step_x, :step_y],
                                                  hexagon[
                                                  i - small_image_h - lateral_mid: i - small_image_h - lateral_mid + step_x,
                                                  j: j + step_y])

                        chosen_image = find_first_diff_neighbor(masked_similar_img,
                                                                [down_left_neigh, top_left_neigh, upper_neigh,
                                                                 upper_right_neigh])

                    #             up = 0 if i - offset == 0 else 1
                    hexagon[i - lateral_mid: i + step_x - lateral_mid, j:j + step_y, :] += np.multiply(
                        mask[:step_x, :step_y],
                        chosen_image)[
                                                                                           :step_x,
                                                                                           :step_y, :]

                side = 1 - side

    hexagon = hexagon[lateral_mid:hex_H - lateral_mid, lateral_mid:hex_W - lateral_mid, :].copy()

    print(hexagon.shape)
    # cv.imwrite('../Hex_obama_100.png', hexagon)
    cv.imshow('asda', hexagon)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return hexagon
