#===============================================================================
# Projeto Desafio da segmentação
# Alunos: 
# Eric Machado - 2191083
# Gabriel Leão Bernarde - 2194228
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

import sys
import numpy as np
import cv2
import math

#===============================================================================

# *** Valores de ajuste ***
# Gamma
GAMMA = 0.7

# Blur
KERNEL = 19
SIGMA = 3.14

# Limiarizacao
T_KERNEL = 3
T_SIGMA = 0

# Blur 2
B2_KERNEL = 9

# Selecao
MAX_SIZE_FROM_MEDIAN = 1.6
MAX_SIZE_FROM_MEAN = 4

def main ():
    gabarito = escolhe_gabarito()

    img = cv2.imread("img/{}.bmp".format(gabarito), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()

    img = gammaCorrection(img, GAMMA)

    cv2.imshow('00 - GAMMA', img)
    cv2.imwrite('00 - GAMMA.png', img)

    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.astype(np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    borrada = cv2.GaussianBlur(img, (KERNEL, KERNEL), SIGMA)

    cv2.imshow('01 - Borrada', borrada)
    cv2.imwrite('01 - Borrada.png', borrada * 255)

    borrada = (borrada * 255).astype(np.uint8)
    binarizada = cv2.adaptiveThreshold(borrada, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, T_KERNEL, T_SIGMA)
    cv2.imshow('02 - Binarizada', binarizada * 255)
    cv2.imwrite('02 - Binarizada.png', binarizada * 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    fechada = cv2.morphologyEx(binarizada, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('03 - FECHADA', fechada * 255)
    cv2.imwrite('03 - FECHADA.png', fechada * 255)

    borrada2 = cv2.medianBlur(fechada, ksize=B2_KERNEL)

    cv2.imshow('04 - Borrada (binarizada)', borrada2 * 255)
    cv2.imwrite('04 - Borrada (binarizada).png', borrada2 * 255)

    aberta = cv2.morphologyEx(borrada2, cv2.MORPH_OPEN, kernel)

    cv2.imshow('05 - ABERTA', aberta * 255)
    cv2.imwrite('05 - ABERTA.png', aberta * 255)

    # Componentes conexos
    _n_components, _labels, stats, _centroids = cv2.connectedComponentsWithStats(aberta, connectivity=4)

    # Desconsidera fundo
    valid_stats = stats[1:]

    mostra_componentes(valid_stats, img_out, '06 - OUT')

    n_graos = conta_grudados(valid_stats)
    diff = abs(gabarito - n_graos)
    erro = diff * 100 / gabarito

    print('Gabarito escolhido:', gabarito)
    print('%d grãos contados.' % n_graos)
    print('Erro: %.2f%%' % erro)

    cv2.waitKey()
    cv2.destroyAllWindows()

def escolhe_gabarito():
    while(True):
        print('1 - 60')
        print('2 - 82')
        print('3 - 114')
        print('4 - 150')
        print('5 - 205')

        escolha = str(input('Escolha uma imagem: '))

        if escolha == '1':
            return 60
        elif escolha == '2':
            return 82
        elif escolha == '3':
            return 114
        elif escolha == '4':
            return 150
        elif escolha == '5':
            return 205
        else:
            print('Opcao invalida.')

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    lookup_table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    lookup_table = np.array(lookup_table, np.uint8)

    return cv2.LUT(src, lookup_table)

def conta_grudados(valid_stats):
    count = 0

    sizes = [stat[cv2.CC_STAT_AREA] for stat in valid_stats]
    sizes_mean = np.mean(sizes)
    soltos = [blob for blob in valid_stats if blob[cv2.CC_STAT_AREA] < sizes_mean * MAX_SIZE_FROM_MEAN]
    rice_sizes = [blob[cv2.CC_STAT_AREA] for blob in soltos]
    print("Soltos: ", len(soltos))
    median_size = np.median(rice_sizes)
    print("Mean: ", median_size)

    for blob in valid_stats:
        if blob[cv2.CC_STAT_AREA] <= median_size * MAX_SIZE_FROM_MEDIAN:
            count += 1
        else:
            count +=  math.ceil(blob[cv2.CC_STAT_AREA] / median_size)

    return count

def mostra_componentes(componentes, img_out, label):
    for stat in componentes:
        top_left_yx = (stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP])
        bottom_right_yx = (stat[cv2.CC_STAT_LEFT] + stat[cv2.CC_STAT_WIDTH], stat[cv2.CC_STAT_TOP] + stat[cv2.CC_STAT_HEIGHT])
        cv2.rectangle(img_out, top_left_yx, bottom_right_yx, (0,0,1))

    cv2.imshow(label, img_out)
    cv2.imwrite("{}.png".format(label), img_out * 255)


if __name__ == '__main__':
    main()

#===============================================================================
