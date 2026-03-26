# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QTextEdit

class SignalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # 1. 界面基础设置
        self.setWindowTitle("张俊熠的信号显示系统 - 实验三")
        self.setGeometry(100, 100, 900, 650)

        # 2. 创建主部件
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # 3. 顶部：图形显示框 (Matplotlib 嵌入)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # 4. 中间：信息显示框 (Text Edit)
        self.info_display = QTextEdit()
        self.info_display.setFixedHeight(80)
        self.info_display.setReadOnly(True)
        self.info_display.setText("系统状态：已启动。请点击下方按钮显示对应的离散信号。")
        self.layout.addWidget(self.info_display)

        # 5. 底部：按钮区域 (Horizontal Layout)
        self.btn_layout = QHBoxLayout()
        self.btn_impulse = QPushButton("显示：单位冲激信号 δ[n]")
        self.btn_step = QPushButton("显示：单位阶跃信号 u[n]")
        self.btn_sine = QPushButton("显示：离散正弦信号")
        
        self.btn_layout.addWidget(self.btn_impulse)
        self.btn_layout.addWidget(self.btn_step)
        self.btn_layout.addWidget(self.btn_sine)
        self.layout.addLayout(self.btn_layout)

        # 6. 绑定按钮点击事件 (信号与槽)
        self.btn_impulse.clicked.connect(self.plot_impulse)
        self.btn_step.clicked.connect(self.plot_step)
        self.btn_sine.clicked.connect(self.plot_sine)

    def clear_canvas(self):
        self.ax.clear()
        self.ax.grid(True, linestyle='--', alpha=0.7)

    def plot_impulse(self):
        """单位冲激信号 δ[n]"""
        self.clear_canvas()
        n = np.arange(-5, 6)
        y = np.where(n == 0, 1, 0)
        self.ax.stem(n, y)
        self.ax.set_title("Unit Impulse Signal: δ[n]")
        self.canvas.draw()
        self.info_display.setText("当前显示：单位冲激信号 δ[n]。该信号仅在 n=0 时幅度为 1，其余为 0。")

    def plot_step(self):
        """单位阶跃信号 u[n]"""
        self.clear_canvas()
        n = np.arange(-5, 11)
        y = np.where(n >= 0, 1, 0)
        self.ax.stem(n, y)
        self.ax.set_title("Unit Step Signal: u[n]")
        self.canvas.draw()
        self.info_display.setText("当前显示：单位阶跃信号 u[n]。该信号在 n>=0 时恒为 1。")

    def plot_sine(self):
        """离散正弦信号"""
        self.clear_canvas()
        n = np.arange(0, 21)
        y = np.sin(0.1 * np.pi * n)
        self.ax.stem(n, y)
        self.ax.set_title("Discrete Sine Signal: sin(0.1πn)")
        self.canvas.draw()
        self.info_display.setText("当前显示：离散正弦信号。展示了正弦波在离散时间点上的采样值。")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignalApp()
    window.show()
    sys.exit(app.exec_())