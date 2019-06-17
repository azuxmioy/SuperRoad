from util import *

class CSV2Image:
    def __init__(self, csv_filename):
        self.csv = csv_filename
        self.h = 16
        self.w = 16
        self.imgwidth = int(math.ceil((600.0/self.w))*self.w)
        self.imgheight = int(math.ceil((600.0/self.h))*self.h)
        self.nc = 3

    # Convert an array of binary labels to a uint8
    def binary_to_uint8(self, img):
        rimg = (img * 255).round().astype(np.uint8)
        return rimg

    # If you want to save the image, run with "save = True"
    def reconstruct_from_labels(self, image_id, save = False):
        im = np.zeros((self.imgwidth, self.imgheight), dtype=np.uint8)
        f = open(self.csv)
        lines = f.readlines()
        image_id_str = '%.3d_' % image_id
        for i in range(1, len(lines)):
            line = lines[i]
            if not image_id_str in line:
                continue
            tokens = line.split(',')
            id = tokens[0]
            prediction = int(tokens[1])
            tokens = id.split('_')
            i = int(tokens[1])
            j = int(tokens[2])
            je = min(j+self.w, self.imgwidth)
            ie = min(i+self.h, self.imgheight)
            if prediction == 0:
                adata = np.zeros((self.w,self.h))
            else:
                adata = np.ones((self.w,self.h))
            im[j:je, i:ie] = self.binary_to_uint8(adata)
        if save:
            Image.fromarray(im).save('prediction_' + '%.3d' % image_id + '.png')
        return im

