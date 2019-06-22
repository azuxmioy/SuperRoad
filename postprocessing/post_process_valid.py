import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.sparse import coo_matrix
from skimage.color import rgb2lab, lab2rgb
from graph_cut import GraphCut
import re

import PIL.Image as Image

data_path = './../dataset/'
sigma = 10
alpha = 2
thres = 0.75
foreground_threshold = 0.5

def get_pairwise(test_image, mask):

    H, W, _ = np.array(test_image).shape

    row = []
    col = []
    data = []

    for y in range(H):
        for x in range(W):

            if( y+1 < H):
                row.append( np.ravel_multi_index((y, x), dims=(H, W)) )
                col.append( np.ravel_multi_index((y+1, x), dims=(H, W)) )
                B =  np.exp( - np.sum( (test_image[y,x,:] - test_image[y+1, x,:])**2) / (2 * sigma**2) )
                data.append(alpha * B)

            if(x+1 < W and y-1 >= 0):
                row.append( np.ravel_multi_index((y, x), dims=(H, W)) )
                col.append( np.ravel_multi_index((y-1, x+1), dims=(H, W)) )
                B =  np.exp( - np.sum( (test_image[y,x,:] - test_image[y-1, x+1,:])**2) / (2 * sigma**2) ) / np.sqrt(2)
                data.append(alpha * B)

            if( x+1 < W):
                row.append( np.ravel_multi_index((y, x), dims=(H, W)) )
                col.append( np.ravel_multi_index((y, x+1), dims=(H, W)) )
                B =  np.exp( - np.sum( (test_image[y,x,:] - test_image[y, x+1,:])**2) / (2 * sigma**2) )
                data.append(alpha * B)

            if(x+1 < W and y+1 <H ):
                row.append( np.ravel_multi_index((y, x), dims=(H, W)) )
                col.append( np.ravel_multi_index((y+1, x+1), dims=(H, W)) )
                B =  np.exp( - np.sum( (test_image[y,x,:] - test_image[y+1, x+1,:])**2) / (2 * sigma**2) ) / np.sqrt(2)
                data.append(alpha * B)

    return coo_matrix((data, (row, col)), shape=(H*W, H*W))

def submit(test_masks, test_input_ids, submission_filename):
    def patch_to_label(patch):
        # percentage of pixels > 1 required to assign a foreground label to a patch
        df = np.mean(patch)
        if df > foreground_threshold:
            return 1
        else:
            return 0
    
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        patch_size = 16
        for idx in range(len(test_masks)):
            im = test_masks[idx]
            img_number = test_input_ids[idx]
            for j in range(0, im.shape[1], patch_size):
                for i in range(0, im.shape[0], patch_size):
                    patch = im[i:i + patch_size, j:j + patch_size]
                    label = patch_to_label(patch)
                    f.writelines("{:03d}_{}_{},{}\n".format(img_number, j, i, label))


if __name__ == '__main__':
    with open('../results/valid_softLabels_31000.pkl', 'rb') as f:
        soft_labels1 = pickle.load(f)
    #with open('../results/softLabels_3500.pkl', 'rb') as f:
    #    soft_labels2 = pickle.load(f)
    #with open('../results/softLabels_31000.pkl', 'rb') as f:
    #    soft_labels3 = pickle.load(f)

    all_test_timage =  os.listdir(data_path + 'valid_input/')
    all_test_image_paths = sorted([ data_path + 'valid_input/' + str(path) for path in all_test_timage])
    all_test_label =  os.listdir(data_path + 'valid_output/')
    all_test_label_paths = sorted([ data_path + 'valid_output/' + str(path) for path in all_test_label])
    
    ids = [int(re.search(r"\d+", path).group(0)) for path in all_test_image_paths]

    test_output = []
    print(len(all_test_image_paths))

    rr = 0
    rn = 0
    nr = 0
    nn = 0

    for i in range (len(all_test_image_paths)):
    #for i in range (1):
        print(i)
        soft_mask = soft_labels1[i]

        #soft_mask = ( soft_labels1[i] + soft_labels3[i]) / 2.0
        
        img = np.array(Image.open(all_test_image_paths[i]))
        label = np.array(Image.open(all_test_label_paths[i])) // 200

        gt_mask = 1000 * np.where(soft_mask > thres, 1, 0 )

        unaries = np.reshape(np.flip(soft_mask + gt_mask, axis=2), [-1, 2])

        img_normalize = img / 255.0
        img_lab = rgb2lab(img_normalize)

        pairwise = get_pairwise(img_lab, soft_mask)

        bk = GraphCut(unaries.shape[0], pairwise.nnz + unaries.shape[0]*2)
        bk.set_unary(unaries)
        bk.set_neighbors(pairwise)
        cost = bk.minimize()
        prediction = bk.get_labeling()

        smooth_mask = np.reshape(prediction, [img.shape[0], img.shape[1]]).astype(int)

        print(smooth_mask.shape)
        print(label.shape)
        
        rr += np.sum((smooth_mask == 1) * (label == 1))
        rn += np.sum((smooth_mask == 1) * (label == 0))
        nr += np.sum((smooth_mask == 0) * (label == 1))
        nn += np.sum((smooth_mask == 0) * (label == 0))
                    

        #test_output.append(smooth_mask)
        #test_output.append(np.argmax(soft_mask, axis=2))
        #plt.subplot(1,2,1)
        #plt.imshow(label)
        #plt.subplot(1,2,2)
        #plt.imshow(smooth_mask)
        #plt.show()
        
        
    mIoU = ( 0.5 * rr / (rr + rn + nr) ) + ( 0.5 * nn / (nn + rn + nr) )
    print('[INFO] Validation mIoU')
    print (mIoU)

