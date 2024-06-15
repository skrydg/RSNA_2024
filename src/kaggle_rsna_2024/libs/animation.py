import matplotlib.pyplot as plt
from matplotlib import animation, rc

from kaggle_rsna_2024.libs.input.input_3d_scan import Input3dScan

class ImageAnimation:
    def __init__(self):
        pass

    def show(self, image, title="image animatin"):
        rc('animation', html='jshtml')
        fig = plt.figure(figsize=(6, 6))
        plt.axis('off')
        im = plt.imshow(image[0], cmap="CMRmap")

        def animate_func(i):
            im.set_array(image[i])
            return [im]
        plt.title(title)
        plt.close()

        return animation.FuncAnimation(fig, animate_func, frames=image.shape[0], interval=1000//10)
    
class ScanAnimation:
    def __init__(self):
        self.image_animation = ImageAnimation()

    def show(self, scan: Input3dScan):
        image = scan.get_image_array()
        title = f'series_id = {scan.series_id}, scan_type={scan.scan_type}'
        return self.image_animation.show(image, title)
        