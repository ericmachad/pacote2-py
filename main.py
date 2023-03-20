# ===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
# -------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Alunos: Eric Machado e Gabriel Leão Bernarde
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

import sys
import timeit
import numpy as np
import cv2

# ===============================================================================

INPUT_IMAGE = 'arroz.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.75
ALTURA_MIN = 19
LARGURA_MIN = 19
N_PIXELS_MIN = 400

# ===============================================================================

ARROZ = -1
BACKGROUND = 0
FOREGROUND = 1


def binariza(img, threshold):
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.

Valor de retorno: versão binarizada da img_in.'''

    # TODO: escreva o código desta função.
    # Dica/desafio: usando a função np.where, dá para fazer a binarização muito
    # rapidamente, e com apenas uma linha de código!

# -------------------------------------------------------------------------------
    return np.where(img > threshold, FOREGROUND, BACKGROUND).astype(np.float32)


def rotula(img, largura_min, altura_min, n_pixels_min):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].

Parâmetros: img: imagem de entrada E saída.
            largura_min: descarta componentes com largura menor que esta.
            altura_min: descarta componentes com altura menor que esta.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
com os seguintes campos:

'label': rótulo do componente.
'n_pixels': número de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
respectivamente: topo, esquerda, baixo e direita.'''

    # TODO: escreva esta função.
    # Use a abordagem com flood fill recursivo.
# ===============================================================================
    altura = len(img)
    largura = len(img[0])
    rotulo = 0.1
    componentes = []

# Separa o background dos pontos possiveis para ser arroz
    img = np.where(img == FOREGROUND, ARROZ, BACKGROUND)

    for i in range(0, altura):
        for j in range(0, largura):
            if img[i][j][0] == ARROZ:
                componente = dict(
                    label=rotulo,
                    n_pixels=0,
                    L=j,
                    T=i,
                    R=j,
                    B=i
                )

                componente_encontrado = inunda(
                    img, altura, largura, i, j, componente)
                
                if componente_encontrado['n_pixels'] >= n_pixels_min:
                    componentes.append(componente_encontrado)
            rotulo += 0.1

    return componentes


def inunda(img, largura, altura, y, x, componente):

    if img[y][x][0] != ARROZ:
        return componente

    img[y][x][0] = componente['label']
    componente['n_pixels'] += 1

# verificar os cantos do componente
    if x < componente['L']:
        componente['L'] = x

    if y > componente['T']:
        componente['T'] = y

    if x > componente['R']:
        componente['R'] = x

    if y < componente['B']:
        componente['B'] = y

# Chama a recursão nos vizinhos

    # Esquerda
    if x > 0:
        componente = inunda(img, altura, largura, y, x - 1, componente)

    # Cima
    if y > 0:
        componente = inunda(img, altura, largura, y - 1, x, componente)

    # Direita
    if x < largura - 1:
        componente = inunda(img, altura, largura, y, x + 1, componente)

    # Baixo
    if y < altura - 1:
        componente = inunda(img, altura, largura, y + 1, x, componente)

    return componente


def main():

    # Abre a imagem em escala de cinza.
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.astype(np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza(img, THRESHOLD)
    cv2.imshow('01 - binarizada', img)
    cv2.imwrite('01 - binarizada.png', img*255)

    start_time = timeit.default_timer()
    componentes = rotula(img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len(componentes)
    print('Tempo: %f' % (timeit.default_timer() - start_time))
    print('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle(img_out, (c['L'], c['T']), (c['R'], c['B']), (0, 0, 1))

    cv2.imshow('02 - out', img_out)
    cv2.imwrite('02 - out.png', img_out*255)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# ===============================================================================
