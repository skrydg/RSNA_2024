import matplotlib.pyplot as plt
from matplotlib import animation, rc

from kaggle_rsna_2024.libs.input.input_3d_scan import Input3dScan

class ScanAnimation:
    def __init__(self):
        pass
    def show(self, scan: Input3dScan):
        image = scan.get_image_array()

        rc('animation', html='jshtml')
        fig = plt.figure(figsize=(6, 6))
        plt.axis('off')
        im = plt.imshow(image[0], cmap="CMRmap")
        text = plt.text(0.05, 0.05, f'Slide {1}', transform=fig.transFigure, fontsize=16, color='darkblue')

        def animate_func(i):
            im.set_array(image[i])
            return [im]
        plt.title(f'id = {id}, series')
        plt.close()

        return animation.FuncAnimation(fig, animate_func, frames=image.shape[0], interval=1000//10)