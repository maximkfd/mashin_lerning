import pylab as pl
from test22 import draw_plane
from test22 import onclick
from test22 import press

if __name__ == '__main__':
    figure = pl.figure()
    figure.canvas.mpl_connect('button_press_event', onclick)
    figure.canvas.mpl_connect('key_press_event', press)
    draw_plane(5)
    while True:
        pl.pause(0.05)
