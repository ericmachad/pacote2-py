#===============================================================================
# Projeto Bloom
# Alunos: 
# Eric Machado - 2191083
# Gabriel Leão Bernarde - 2194228
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

THRESHOLD = .55
VALINI_SIGMA = 10
VALINI_KERNEL = 3

REPETIÇÕES_FILTRO_GAUSSIANO = 3
REPETIÇÕES_FILTRO_MEDIA = 40

INPUT_IMG = 'GT2.bmp'
#INPUT_IMG = 'Wind Waker GC.bmp'

IMG_MULT = 0.9
BLOOM_MULT = .15

def bloom_gaussiano(img):
    bloom = np.zeros(img.shape)

    sigma = VALINI_SIGMA

    for _i in range(0, REPETIÇÕES_FILTRO_GAUSSIANO):
        bloom += cv2.GaussianBlur(img, (0, 0), sigma)
        sigma *= 2

    return bloom

def bloom_media(img):
    bloom = np.zeros(img.shape)
    kernel = VALINI_KERNEL
    for _i in range(0, REPETIÇÕES_FILTRO_GAUSSIANO):
        bloom_media = cv2.blur(img, (kernel, kernel))
        for _j in range(0, REPETIÇÕES_FILTRO_MEDIA - 1):
            bloom_media = cv2.blur(bloom_media, (kernel, kernel))
        bloom += bloom_media
        kernel+=3

    return bloom

def main():
    img = cv2.imread(INPUT_IMG)
    if img is None:
        print('Erro ao abrir a imagem.\n')
        sys.exit()
    # Convertendo para float32.
    img = img.astype(np.float32) / 255
    # Criando uma máscara em escala de cinza e convertendo para 3 canais, para poder operar.
    MASK_CINZA = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    MASK_CINZA = MASK_CINZA.reshape(MASK_CINZA.shape[0], MASK_CINZA.shape[1], 1)
    
    cv2.imshow('01 - Original', img)

    img_limiar = np.where(MASK_CINZA > THRESHOLD, img, 0)

    cv2.imshow('02 - limiar', img_limiar)

    bloom = bloom_gaussiano(img_limiar)

    cv2.imshow('03 - Bloom Filtro Gaussiano', bloom)
    cv2.imwrite('03 - Bloom Filtro Gaussiano.bmp', bloom * 255)

    bloom_media = bloom_media(img_limiar)

    cv2.imshow('04 - Bloom Media', bloom_media)
    cv2.imwrite('04 - Bloom Media.bmp', bloom_media * 255)

    img_out = img * IMG_MULT + bloom * BLOOM_MULT
    img_out_mean = img * IMG_MULT + bloom_media * BLOOM_MULT

    cv2.imshow('05 - OUT Gaussiano', img_out)
    cv2.imwrite('05 - OUT Gaussiano.bmp', img_out * 255)

    cv2.imshow('06 - OUT Media', img_out_mean)
    cv2.imwrite('06 - OUT Media.bmp', img_out_mean * 255)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
