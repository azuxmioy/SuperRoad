import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.sparse import coo_matrix
from skimage.color import rgb2lab, lab2rgb
from graph_cut import GraphCut
import re
import math
from tqdm import tqdm

import PIL.Image as Image

data_path = './../dataset/'
sigma = 10
alpha = 2
thres = 0.75
foreground_threshold = 0.5

def sigmoid(x, t):
    return 1 / (1 + math.exp(-t * x))

def convert_mask_to_weird_format(mask):
    ret_mask = np.zeros((mask.shape[0], mask.shape[1], 2))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            ret_mask[i][j][0] = mask[i][j]
            ret_mask[i][j][1] = 1 - ret_mask[i][j][0]
    return ret_mask

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


def smoothing(mask):
    window = 16
    lookahead = 3
    minthreshold = 0.25
    sigmoidbandwidth = 20
    assert window >= lookahead

    new_mask = np.zeros_like(mask)
    for i in tqdm(range(mask.shape[0])):
        for j in range(mask.shape[1]):
            threshold_val = []
            max_val = []
            for direction in [1,-1]:
                # lookahead
                nbr_val = []
                nbr_cnt = 0
                for r in range(0, lookahead+1,1):
                    if 0 <= direction*r+i and direction*r+i < mask.shape[0]:
                        nbr_val.append(mask[i+direction*r,j,1])
                        nbr_cnt += 1
                    else:
                        nbr_val.append(0)
                threshold_val.append(sum(nbr_val) / nbr_cnt)

                # window
                nbr_val = []
                nbr_cnt = 0
                for r in range(0, window+1,1):
                    if 0 <= direction*r+i and direction*r+i < mask.shape[0]:
                        nbr_val.append(mask[i+direction*r,j,1])
                        nbr_cnt += 1
                    else:
                        nbr_val.append(0)
                max_val.append(max(nbr_val))

            # do changes only if min_val is bigger than threshold (otherwise we are at the border of the road)
            max_val[0] *= (threshold_val[0] >= minthreshold) #sigmoid(threshold_val[0] - minthreshold, sigmoidbandwidth)
            max_val[1] *= (threshold_val[1] >= minthreshold) #sigmoid(threshold_val[1] - minthreshold, sigmoidbandwidth)

            # take maximum from the previous value and the minimum from both directions
            new_val = np.clip(min(max_val[0], max_val[1]), 0, 1)
            new_mask[i,j,1] = max(mask[i,j,1], new_val)
            new_mask[i,j,0] = 1 - new_mask[i,j,1]
    return new_mask

def anglesmoothing(mask, angle):
    if angle == 90 or angle == -90:
        return smoothing(mask.transpose(1,0,2)).transpose(1,0,2)
        
    #embed to bigger mask
    big_mask = np.zeros((mask.shape[0]*2, mask.shape[1]*2))
    big_mask[
        int(mask.shape[0]/2) : int(mask.shape[0]+mask.shape[0]/2), 
        int(mask.shape[1]/2) : int(mask.shape[1]+mask.shape[1]/2)] = mask[:, :, 0]
    
    #convert to image and rotate
    im = Image.fromarray(np.uint8(big_mask*255))
    im = im.rotate(angle)
    rotated_mask = np.array(im) / 255.0

    #smoothing
    smoothed_mask = smoothing(convert_mask_to_weird_format(rotated_mask))[:,:,0]
    #smoothed_mask = convert_mask_to_weird_format(rotated_mask)[:,:,0]
    
    #rotate back
    im = Image.fromarray(np.uint8(smoothed_mask*255))
    im = im.rotate(-angle)
    smoothed_mask = np.array(im) / 255.0
    
    #crop to original size
    offset = (int(round((smoothed_mask.shape[0]-mask.shape[0])/2)),
                int(round((smoothed_mask.shape[1]-mask.shape[1])/2)))
    cropped_mask = smoothed_mask[offset[0] : offset[0]+mask.shape[0], 
                offset[1] : offset[1]+mask.shape[1]]
    
    return convert_mask_to_weird_format(cropped_mask)

if __name__ == '__main__':
    with open('../results/valid_softLabels_31000.pkl', 'rb') as f:
        soft_labels1 = pickle.load(f)
    with open('../results/valid_softLabels_21000.pkl', 'rb') as f:
        soft_labels2 = pickle.load(f)

    all_test_timage =  os.listdir(data_path + 'valid_input/')
    all_test_image_paths = sorted([ data_path + 'valid_input/' + str(path) for path in all_test_timage])
    all_test_label =  os.listdir(data_path + 'valid_output/')
    all_test_label_paths = sorted([ data_path + 'valid_output/' + str(path) for path in all_test_label])
    ids = [int(re.search(r"\d+", path).group(0)) for path in all_test_image_paths]

    test_output = []
    rr = 0
    rn = 0
    nr = 0
    nn = 0

    for i in range (len(all_test_image_paths)):
        print("Image number: " + str(i))
        print(all_test_image_paths[i])
        

        soft_mask = ( soft_labels1[i] + soft_labels2[i]) / 2.0

        #soft_mask = (soft_labels1[i] + soft_labels2[i] + soft_labels3[i]) / 3.0
        #out_mask = soft_mask 

        soft_mask_0 = smoothing(soft_mask)
        soft_mask_90 = anglesmoothing(soft_mask, 90)
        soft_mask_45 = anglesmoothing(soft_mask, 45)
        soft_mask_315 = anglesmoothing(soft_mask, -45)       
        out_mask = soft_mask
        out_mask = np.maximum(out_mask, soft_mask_0)
        out_mask = np.maximum(out_mask, soft_mask_90)
        out_mask = np.maximum(out_mask, soft_mask_45)
        out_mask = np.maximum(out_mask, soft_mask_315)

        img = np.array(Image.open(all_test_image_paths[i]))
        label = np.array(Image.open(all_test_label_paths[i])) // 200

        gt_mask = 1000 * np.where(out_mask > thres, 1, 0 )
        unaries = np.reshape(np.flip(out_mask + gt_mask, axis=2), [-1, 2])

        img_normalize = img / 255.0
        img_lab = rgb2lab(img_normalize)

        pairwise = get_pairwise(img_lab, out_mask)
        bk = GraphCut(unaries.shape[0], pairwise.nnz + unaries.shape[0]*2)
        bk.set_unary(unaries)
        bk.set_neighbors(pairwise)
        cost = bk.minimize()
        prediction = bk.get_labeling()
        
        smooth_mask = np.reshape(prediction, [img.shape[0], img.shape[1]]).astype(int)
        test_output.append(smooth_mask)

        rr += np.sum((smooth_mask == 1) * (label == 1))
        rn += np.sum((smooth_mask == 1) * (label == 0))
        nr += np.sum((smooth_mask == 0) * (label == 1))
        nn += np.sum((smooth_mask == 0) * (label == 0))

        '''
        plt.subplot(1,3,1)
        plt.imshow(soft_mask[:,:,1], cmap = "Greys", interpolation='None')
        plt.subplot(1,3,2)
        plt.imshow(out_mask[:,:,1], cmap = "Greys", interpolation='None')
        plt.subplot(1,3,3)
        plt.imshow(smooth_mask)
        fname = all_test_image_paths[i].split('/')[-1].split('.')[0]
        plt.savefig("smooth0/" + fname + "_out" + ".png")
        '''
        
    mIoU = ( 0.5 * rr / (rr + rn + nr) ) + ( 0.5 * nn / (nn + rn + nr) )
    print('[INFO] Validation mIoU')
    print (mIoU)

