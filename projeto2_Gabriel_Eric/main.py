# Projeto Filtro da média
# Alunos: 
# Eric Machado - 2191083
# Gabriel Leão Bernarde - 2194228
# Universidade Tecnológica Federal do Paraná
# ===============================================================================
import sys
import timeit
import numpy as np
import cv2

ALTURA_JANELA = 3
LARGURA_JANELA = 13
IMAGEM_ENTRADA = 'b'

def imagem_integral(img):
    altura, largura, n_canais = img.shape
    
    integral = np.zeros(img.shape)

    for canal in range(0, n_canais):
        for y in range(0, altura):
            integral[y][0][canal] = img[y][0][canal]
            for x in range(1, largura):
                integral[y][x][canal] = img[y][x][canal] + integral[y][x - 1][canal]

        for y in range(1, altura):
            for x in range(0, largura):
                integral[y][x][canal] = integral[y][x][canal] + integral[y - 1][x][canal]

    return integral


def filtro_img_integral(img, altura_janela, largura_janela):
    altura, largura, n_canais = img.shape
    img_out = np.zeros(img.shape)
    integral = imagem_integral(img)
    for canal in range(0, n_canais):
        for y in range(0,altura):
            for x in range(0,largura):
                divisor_altura=altura_janela
                divisor_largura=largura_janela
                topo_esquerda = y - altura_janela // 2
                topo_direita = y + altura_janela // 2
                baixo_esquerda = x - largura_janela // 2
                baixo_direita = x + largura_janela // 2
                if(topo_esquerda < 0):
                    divisor_altura += topo_esquerda
                    topo_esquerda = 0
                if(baixo_esquerda < 0):
                    divisor_largura += baixo_esquerda
                    baixo_esquerda = 0
                if(topo_direita > altura-1):
                    divisor_altura -= topo_direita - altura + 1
                    topo_direita = altura-1
                if(baixo_direita > largura-1):
                    divisor_largura -= baixo_direita - largura + 1
                    baixo_direita = largura-1

                img_out[y][x][canal] = integral[topo_direita][baixo_direita][canal]
                if(topo_esquerda != 0):
                    img_out[y][x][canal] -= integral[topo_esquerda-1][baixo_direita][canal]
                if(baixo_esquerda != 0):
                    img_out[y][x][canal] -= integral[topo_direita][baixo_esquerda-1][canal]
                if(baixo_esquerda != 0 and topo_esquerda != 0):
                    img_out[y][x][canal] += integral[topo_esquerda-1][baixo_esquerda-1][canal]
                img_out[y][x][canal] = img_out[y][x][canal] / (divisor_largura * divisor_altura)
                

    return img_out   


def calcula_media(janela):
    altura, largura = janela.shape

    soma = 0
    for y in range(0, altura):
        for x in range(0, largura):
            soma += janela[y][x]

    return soma / (largura * altura)

def filtro_ingenuo(img, altura_janela, largura_janela):
    altura, largura, n_canais = img.shape

    img_out = np.zeros(img.shape)

    for c in range(0, n_canais):
        for y in range(altura_janela // 2, altura - altura_janela // 2):
            start_y = y - altura_janela // 2
            end_y = max(y + altura_janela // 2, 1) + 1
            for x in range(largura_janela // 2, largura - largura_janela // 2):
                start_x = x - largura_janela // 2
                end_x = max(x + largura_janela // 2, 1) + 1

                img_out[y][x][c] = calcula_media(img[start_y:end_y, start_x:end_x, c])

    return img_out              

def filtro_separavel(img,altura_janela,largura_janela):
    return filtro_ingenuo(filtro_ingenuo(img, altura_janela, 1), 1, largura_janela)

def main():
    img = cv2.imread("Exemplos/{} - Original.bmp".format(IMAGEM_ENTRADA))
    if img is None:
        print('Erro ao abrir a imagem.\n')
        sys.exit()

    img = img.astype(np.float32) / 255
    cv2.imshow('01 - Original', img)
    cv2.imwrite('01 - Original.bmp', img * 255)
    start_time = timeit.default_timer()
    img = filtro_ingenuo(img, ALTURA_JANELA, LARGURA_JANELA)
    print('Tempo: %f' % (timeit.default_timer() - start_time))

    cv2.imshow('02 - out', img)
    cv2.imwrite('02 - out.png', img * 255)
    

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()