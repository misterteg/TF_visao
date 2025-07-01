import sys, os
import tkinter as tk
import numpy as np
import cv2  # Importar OpenCV

from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsPathItem, QGraphicsEllipseItem, QWidget # Importar QWidget para usar QKeyEvent
from PyQt5.QtGui import QPixmap, QPainterPath, QPen, QColor, QPainter
from PyQt5.QtCore import Qt, QPointF, QEvent # Importar QEvent para filtrar eventos

os.environ["QT_QPA_PLATFORM"] = "xcb"

# --- Função para gerar os mapas de remapeamento ---
# (Mantida como uma função externa por clareza, mas pode ser um método de ImageViewer)
def gerar_mapas_remap(malha_float, nova_W_pyqt, nova_H_pyqt, orig_W_cv, orig_H_cv, N_cols=12, N_rows=12):
    """
    Gera os mapas de coordenadas X e Y para cv2.remap.

    Args:
        malha_float (list): Lista de listas de tuplas (u_red, v_red) representando a malha
                            gerada pelo PyQt, onde (u_red, v_red) são coordenadas na imagem redimensionada pelo PyQt.
        nova_W_pyqt (int): Largura da imagem redimensionada no PyQt.
        nova_H_pyqt (int): Altura da imagem redimensionada no PyQt.
        orig_W_cv (int): Largura da imagem original (carregada pelo OpenCV).
        orig_H_cv (int): Altura da imagem original (carregada pelo OpenCV).
        N_cols (int): Número de colunas na malha (geralmente 12).
        N_rows (int): Número de linhas na malha (geralmente 12).

    Returns:
        tuple: (map_x, map_y) - os dois mapas de coordenadas para cv2.remap.
    """
    # Cria mapas de destino com as dimensões da imagem ORIGINAL (a que vamos deformar)
    map_x = np.zeros((orig_H_cv, orig_W_cv), dtype=np.float32)
    map_y = np.zeros((orig_H_cv, orig_W_cv), dtype=np.float32)

    # Iterar sobre cada pixel (u_out, v_out) na imagem de SAÍDA que queremos gerar
    # (que terá as dimensões da imagem original: orig_W_cv, orig_H_cv)
    for v_out in range(orig_H_cv):
        for u_out in range(orig_W_cv):
            # 1. Mapear a posição (u_out, v_out) na imagem de SAÍDA para os índices na malha de controle
            # Normaliza as coordenadas na grade da malha (0 a N-1)
            # Se a malha tem N colunas, os índices vão de 0 a N-1.
            # Mapeamos u_out (0 a orig_W_cv-1) para i_float (0 a N_cols-1).
            i_float = u_out * (N_cols - 1) / orig_W_cv
            j_float = v_out * (N_rows - 1) / orig_H_cv

            # Pega os índices inteiros e as frações para interpolação bilinear
            i_int = int(i_float)
            j_int = int(j_float)
            fx = i_float - i_int # Fração horizontal
            fy = j_float - j_int # Fração vertical

            # Clamp para evitar acessos fora dos limites da malha (importante!)
            # i_int vai de 0 a N_cols-2 para poder usar i_int+1
            i_int = np.clip(i_int, 0, N_cols - 2)
            j_int = np.clip(j_int, 0, N_rows - 2)

            # 2. Obter os 4 pontos da malha mais próximos (ajustados para as dimensões ORIGINAIS)
            # Os pontos na malha (u_red, v_red) são relativos à imagem redimensionada pelo PyQt.
            # Precisamos convertê-los para coordenadas correspondentes na imagem ORIGINAL.
            # u_red_orig = u_red * orig_W_cv / nova_W_pyqt
            # v_red_orig = v_red * orig_H_cv / nova_H_pyqt

            # Ponto 00 da célula da malha (relativo a j_int, i_int)
            u00_red, v00_red = malha_float[j_int][i_int]
            x00 = u00_red * orig_W_cv / nova_W_pyqt
            y00 = v00_red * orig_H_cv / nova_H_pyqt

            # Ponto 10 da célula da malha (relativo a j_int, i_int+1)
            u10_red, v10_red = malha_float[j_int][i_int + 1]
            x10 = u10_red * orig_W_cv / nova_W_pyqt
            y10 = v10_red * orig_H_cv / nova_H_pyqt

            # Ponto 01 da célula da malha (relativo a j_int+1, i_int)
            u01_red, v01_red = malha_float[j_int + 1][i_int]
            x01 = u01_red * orig_W_cv / nova_W_pyqt
            y01 = v01_red * orig_H_cv / nova_H_pyqt

            # Ponto 11 da célula da malha (relativo a j_int+1, i_int+1)
            u11_red, v11_red = malha_float[j_int + 1][i_int + 1]
            x11 = u11_red * orig_W_cv / nova_W_pyqt
            y11 = v11_red * orig_H_cv / nova_H_pyqt

            # 3. Interpolar bilinearmente para encontrar o ponto (x, y) na imagem original
            # Interpolação na direção 'i' (horizontal - fx)
            x_top = x00 * (1 - fx) + x10 * fx
            y_top = y00 * (1 - fx) + y10 * fx
            x_bottom = x01 * (1 - fx) + x11 * fx
            y_bottom = y01 * (1 - fx) + y11 * fx

            # Interpolação na direção 'j' (vertical - fy)
            x = x_top * (1 - fy) + x_bottom * fy
            y = y_top * (1 - fy) + y_bottom * fy

            # Armazena as coordenadas (x, y) no mapa correspondente ao pixel de saída (u_out, v_out)
            # Lembre-se que OpenCV usa (linha, coluna) -> (v, u)
            map_x[v_out, u_out] = x
            map_y[v_out, u_out] = y

    return map_x, map_y

# --- Classes PyQt modificadas ---

class ControlPoint(QGraphicsEllipseItem):
    def __init__(self, ponto, idx, parent=None):
        super().__init__(-5, -5, 10, 10, parent)
        self.idx = idx
        if idx in [0, 3, 6, 9]:
            self.setBrush(QColor("yellow"))  # quinas
        else:
            self.setBrush(QColor("green"))   # handles
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable)
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges)
        self.setZValue(2)
        self.setPos(ponto)

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionChange and self.parentItem():
            parent = self.parentItem()
            if self.idx % 3 == 0 and len(parent.pontos) == 12:
                delta = value - self.pos()
                idx1 = (self.idx + 1) % 12
                idx2 = (self.idx - 1) % 12
                parent.pontos[idx1].moveBy(delta.x(), delta.y())
                parent.pontos[idx2].moveBy(delta.x(), delta.y())
            parent.updatePath()
        return super().itemChange(change, value)

class BezierQuad(QGraphicsPathItem):
    def __init__(self, pontos):
        super().__init__()
        self.pontos = []
        for i, p in enumerate(pontos):
            ponto = ControlPoint(p, i, self)
            self.pontos.append(ponto)
        self.updatePath()

    def updatePath(self):
        if len(self.pontos) != 12:
            return
        path = QPainterPath()
        path.moveTo(self.pontos[0].pos())
        path.cubicTo(self.pontos[1].pos(), self.pontos[2].pos(), self.pontos[3].pos())
        path.cubicTo(self.pontos[4].pos(), self.pontos[5].pos(), self.pontos[6].pos())
        path.cubicTo(self.pontos[7].pos(), self.pontos[8].pos(), self.pontos[9].pos())
        path.cubicTo(self.pontos[10].pos(), self.pontos[11].pos(), self.pontos[0].pos())
        self.setPath(path)

        # Notifica a view para desenhar a malha (se existir)
        if self.scene() and self.scene().views():
            viewer = self.scene().views()[0]
            if hasattr(viewer, "desenhar_malha"):
                viewer.desenhar_malha() # Chama o método para redesenhar a malha de pontos

    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)
        pen = QPen(QColor("green"), 1, Qt.DashLine)
        painter.setPen(pen)
        for i in range(0, 12, 3):
            quina_atual = self.pontos[i].pos()
            quina_proxima = self.pontos[(i + 3) % 12].pos()
            handle1 = self.pontos[i + 1].pos()
            handle2 = self.pontos[i + 2].pos()
            painter.drawLine(quina_atual, handle1)
            painter.drawLine(quina_proxima, handle2)

class ImageViewer(QGraphicsView):
    def __init__(self, image_path):
        super().__init__()

        self.image_path = image_path # Salva o caminho da imagem original
        self.original_pixmap = QPixmap(image_path) # Carrega a imagem original
        if self.original_pixmap.isNull():
            print(f"Erro: Não foi possível carregar a imagem em {image_path}")
            sys.exit(1)

        # --- Obter dimensões da imagem original (para OpenCV) ---
        # Precisamos do tamanho da imagem *antes* do redimensionamento do PyQt
        self.orig_W_cv = self.original_pixmap.width()
        self.orig_H_cv = self.original_pixmap.height()

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # --- Escalonamento responsivo para a exibição no PyQt ---
        root = tk.Tk() # Inicializa Tkinter para obter informações da tela
        largura_tela, altura_tela = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy() # Destroi a janela Tkinter temporária

        # Calcula a escala para caber a imagem em 2/3 da tela, mantendo a proporção
        escala = min((largura_tela * 4/5) / self.orig_W_cv, (altura_tela * 2/3) / self.orig_H_cv, 1)
        self.nova_W_pyqt = int(self.orig_W_cv * escala) # Armazena a largura redimensionada
        self.nova_H_pyqt = int(self.orig_H_cv * escala) # Armazena a altura redimensionada

        # Redimensiona a imagem para exibição no PyQt
        pixmap_redimensionado = self.original_pixmap.scaled(self.nova_W_pyqt, self.nova_H_pyqt, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.scene.addPixmap(pixmap_redimensionado) # Adiciona a imagem redimensionada à cena

        cx, cy = self.nova_W_pyqt / 2, self.nova_H_pyqt / 2
        offset = min(self.nova_W_pyqt, self.nova_H_pyqt) / 4

        # --- Pontos iniciais para a malha Bézier (baseados nas dimensões redimensionadas) ---
        pontos_iniciais = [
            QPointF(cx - offset, cy - offset),  # 0: topo-esquerdo
            QPointF(cx, cy - offset * 1.3),
            QPointF(cx, cy - offset * 1.3),
            QPointF(cx + offset, cy - offset),  # topo-direito
            QPointF(cx + offset * 1.3, cy),
            QPointF(cx + offset * 1.3, cy),
            QPointF(cx + offset, cy + offset),  # baixo-direito
            QPointF(cx, cy + offset * 1.3),
            QPointF(cx, cy + offset * 1.3),
            QPointF(cx - offset, cy + offset),  # baixo-esquerdo
            QPointF(cx - offset * 1.3, cy),
            QPointF(cx - offset * 1.3, cy)
        ]

        self.quad = BezierQuad(pontos_iniciais)
        self.scene.addItem(self.quad)

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setWindowTitle('Seleção Bézier Ajustável - Pressione ENTER para Processar')
        self.setGeometry(100, 100, self.nova_W_pyqt, self.nova_H_pyqt)

        self.desenhar_malha() # Desenha a malha inicial

    def ponto_bezier(self, p0, p1, p2, p3, t):
        return (
            (1 - t)**3 * p0 +
            3 * (1 - t)**2 * t * p1 +
            3 * (1 - t) * t**2 * p2 +
            t**3 * p3
        )

    def amostrar_bordas(self):
        pontos_borda = []
        for i in range(0, 12, 3):
            p0 = self.quad.pontos[i].pos()
            p1 = self.quad.pontos[i + 1].pos()
            p2 = self.quad.pontos[i + 2].pos()
            p3 = self.quad.pontos[(i + 3) % 12].pos()
            amostrados = [self.ponto_bezier(p0, p1, p2, p3, t) for t in np.linspace(0, 1, 12)]
            pontos_borda.append(amostrados)
        return pontos_borda

    def gerar_malha_12x12(self):
        # Reutiliza as funções internas para manter a organização
        def ponto_bezier(p0, p1, p2, p3, t):
            return (
                (1 - t) ** 3 * p0 +
                3 * (1 - t) ** 2 * t * p1 +
                3 * (1 - t) * t ** 2 * p2 +
                t ** 3 * p3
            )

        def amostrar_curva(curva, n=12):
            p0, p1, p2, p3 = curva
            return [ponto_bezier(p0, p1, p2, p3, t) for t in np.linspace(0, 1, n)]

        curvas = []
        for i in range(0, 12, 3):
            p0 = self.quad.pontos[i].pos()
            p1 = self.quad.pontos[i + 1].pos()
            p2 = self.quad.pontos[i + 2].pos()
            p3 = self.quad.pontos[(i + 3) % 12].pos()
            curvas.append((p0, p1, p2, p3))

        curva_topo, curva_direita, curva_base, curva_esquerda = curvas

        curva_esquerda = tuple(reversed(curva_esquerda))
        curva_base = tuple(reversed(curva_base))

        amostras_topo = amostrar_curva(curva_topo, 12)
        amostras_base = amostrar_curva(curva_base, 12)

        malha = []
        for j in range(12):
            alpha = j / 11.0 # Usar float para a divisão
            linha = []
            for i in range(12):
                pt_top = amostras_topo[i]
                pt_bot = amostras_base[i]
                pt = pt_top * (1 - alpha) + pt_bot * alpha
                linha.append(pt)
            malha.append(linha)

        return malha

    def desenhar_malha(self):
        malha = self.gerar_malha_12x12()
        if not hasattr(self, "pontos_malha"):
            self.pontos_malha = []
            for i in range(12): # Coluna (horizontal)
                linha_pontos = []
                for j in range(12): # Linha (vertical)
                    pt = malha[j][i] # Lembre-se que malha[j][i] é para linha j, coluna i
                    item = QGraphicsEllipseItem(-1.5, -1.5, 3, 3)
                    item.setPos(pt)
                    item.setBrush(QColor("black"))
                    item.setZValue(1)
                    self.scene.addItem(item)
                    linha_pontos.append(item)
                self.pontos_malha.append(linha_pontos)
        else:
            for i in range(12): # Coluna
                for j in range(12): # Linha
                    self.pontos_malha[i][j].setPos(malha[j][i]) # Atualiza posições

    # --- Novo método para processar a malha e aplicar o remapeamento ---
    def processar_remapeamento(self):
        print("Processando remapeamento...")
        # 1. Obter a malha final
        malha_pyqt = self.gerar_malha_12x12() # Retorna lista de listas de QPointF

        # Converter QPointF para tuplas de float (u_red, v_red)
        malha_float = [[(pt.x(), pt.y()) for pt in linha] for linha in malha_pyqt]

        # 2. Gerar os mapas para cv2.remap
        map_x, map_y = gerar_mapas_remap(
            malha_float,
            self.nova_W_pyqt, self.nova_H_pyqt,
            self.orig_W_cv, self.orig_H_cv
        )

        # 3. Carregar a imagem original usando OpenCV
        imagem_original_cv = cv2.imread(self.image_path)
        if imagem_original_cv is None:
            print(f"Erro OpenCV: Não foi possível carregar a imagem em {self.image_path}")
            return

        # 4. Aplicar cv2.remap
        imagem_corrigida = cv2.remap(imagem_original_cv, map_x, map_y, cv2.INTER_LINEAR)

        # 5. Exibir e Salvar a imagem corrigida
        cv2.imshow("Imagem Original", imagem_original_cv)
        cv2.imshow("Imagem Corrigida", imagem_corrigida)

        # Salvar a imagem corrigida
        try:
            base, ext = os.path.splitext(self.image_path)
            nome_salvar = f"{base}_corrigida{ext}"
            cv2.imwrite(nome_salvar, imagem_corrigida)
            print(f"Imagem corrigida salva como: {nome_salvar}")
        except Exception as e:
            print(f"Erro ao salvar a imagem corrigida: {e}")

        cv2.waitKey(0) # Mantém as janelas do OpenCV abertas até uma tecla ser pressionada
        cv2.destroyAllWindows() # Fecha as janelas do OpenCV

    # --- Captura de eventos de teclado ---
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            # Se a tecla Enter for pressionada, processa o remapeamento
            self.processar_remapeamento()
        else:
            # Deixa o comportamento padrão para outras teclas
            super().keyPressEvent(event)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Uso: python seu_script.py <caminho_para_a_imagem>")
        sys.exit(1)

    image_file_path = sys.argv[1]

    app = QApplication(sys.argv)
    viewer = ImageViewer(image_file_path)
    viewer.show()
    sys.exit(app.exec_())
