import sys
import timeit
import numpy as np
import cv2

ALTURA_JANELA = 3
LARGURA_JANELA = 3
IMAGEM_ENTRADA = 'b'

def imagem_integral(img):
    altura, largura, _ = img.shape
    
    integral = img.copy()

    for y in range(0, altura):
        for x in range(1, largura):
            integral[y][x] += integral[y][x - 1]

    for y in range(1, altura):
        for x in range(0, largura):
            integral[y][x] += integral[y - 1][x]

    return integral


def filtro_img_integral(img, altura_janela, largura_janela):
    altura, largura, _ = img.shape

    img_out = img.copy()
    integral = imagem_integral(img)

    for y in range(altura_janela // 2 , altura - altura_janela // 2):
        for x in range(largura_janela // 2, largura - largura_janela // 2):
            topo_esquerda = integral[y - altura_janela // 2 - 1][x - largura_janela // 2 - 1]
            topo_direita = integral[y - altura_janela // 2 - 1][x + largura_janela // 2]
            baixo_esquerda = integral[y + altura_janela // 2][x - largura_janela // 2 - 1]
            baixo_direita = integral[y + altura_janela // 2][x + largura_janela // 2]
                                
            img_out[y][x] = (baixo_direita - topo_direita - baixo_esquerda + topo_esquerda) / (altura_janela * largura_janela)
     
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

def integral(img, h, w):

    ih, iw, _ = img.shape
  
    buffer = img.copy()
    saida = img.copy()

    limH = h//2
    limW = w//2

    for y in range (0, ih):
        for x in range (0, iw):
            if(y != 0):
                buffer[y][x] += buffer[y-1][x]
            if(x != 0):
                buffer[y][x] += buffer[y][x-1]
            if(y != 0 and x != 0):
                buffer[y][x] -= buffer[y-1][x-1]
    
    
    for y in range (0, ih):
        for x in range (0, iw):
            divy = h
            divx = w
            ley = y-limH
            ldy = y+limH
            lex = x-limW
            ldx = x+limW

            if(ley < 0):
                divy += ley
                ley = 0
            if(lex < 0):
                divx += lex
                lex = 0
            if(ldy > ih-1):
                divy -= ldy - ih + 1
                ldy = ih-1
            if(ldx > iw-1):
                divx -= ldx - iw + 1
                ldx = iw-1

            saida[y][x] = buffer[ldy][ldx]
            if(ley != 0):
                saida[y][x] -= buffer[ley-1][ldx]
            if(lex != 0):
                saida[y][x] -= buffer[ldy][lex-1]
            if(lex != 0 and ley != 0):
                saida[y][x] += buffer[ley-1][lex-1]
            saida[y][x] = saida[y][x] / (divx * divy)

    return saida
def separavel(img, h, w):

    ih, iw, _ = img.shape
  
    buffer = img.copy()
    saida = img.copy()

    limH = h//2
    limW = w//2

    for y in range (limH, ih - limH):
        for x in range (limW, iw - limW):
            cont = 0
            for a in range(-limW, limW+1):
                cont += img[y][x+a]
            buffer[y][x] = cont / 3

    for y in range (limH, ih - limH):
        for x in range (limW, iw - limW):
            cont = 0
            for a in range(-limH, limH+1):
                cont += buffer[y+a][x]
            saida[y][x] = cont / 3
    
    return saida
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