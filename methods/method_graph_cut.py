import sys
sys.path.append("../")

from util import *
from methods.method import Method
from methods.graph_cut import GraphCut
from dataset import Dataset
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.sparse import coo_matrix


class Graph_cut(Method):

    def __init__(self):
        self.patch_size = 16
        self.label_thres = 0.50
        self.hist_res = 32
        self.lam = 10

    def train(self, train_input, train_output):

        self.cache_dirname = os.path.join(os.getcwd(), "training_model")
        self.cache_filename = os.path.join(self.cache_dirname, "histogram_cache.pkl")

        if os.path.exists(self.cache_filename):
            print("[Info]: Load from histogram cache!")
            data = self.load_histogram_from_cache()
            self.hist0 = data["hist0"]
            self.hist1 = data["hist1"]
        else:

            print("Constructing color histogram")
            self.hist0, self.hist1 = self.__get_color_histogram(train_input, train_output)
            print("Color histogram done")
            self.create_histogram_cache()


        

    def test(self, test_input, test_input_ids):
        print (len(test_input))

        test_output = []
        fig, ax = plt.subplots()

        for i in range (0,len(test_input)):
            print("testing %d image" % i)

            img = test_input[i]

            unaries =  self.__get_unaries(img, self.hist0, self.hist1)
            pairwise = self.__get_pairwise(img)

            bk = GraphCut(unaries.shape[0], pairwise.nnz + unaries.shape[0]*2)

            print("Compute unaries")

            bk.set_unary(unaries)

            print("Compute pairwise")

            bk.set_neighbors(pairwise)
            
            cost = bk.minimize()
            prediction = bk.get_labeling()
            print(prediction.shape)
            prediction = np.reshape(prediction, [img.shape[0], img.shape[1]]).astype(int)
            print(prediction.shape)
            test_output.append(prediction)
            ax.cla()
            ax.imshow(prediction, cmap = 'gray')
            ax.set_title("image {}".format(test_input_ids[i]))
            plt.pause(0.1)




        '''
        for i in tqdm(range(len(test_input)), "Running test"):
            im = test_input[i]
            h, w, _ = im.shape
            prediction = np.zeros((h, w))
            test_output.append(prediction)
        '''
        return test_output

    # May want to consider smarter ways than just taking threshold
    def patch_to_label(self, patch):
        # percentage of pixels > 1 required to assign a foreground label to a patch
        foreground_threshold = self.label_thres
        df = np.mean(patch)
        if df > foreground_threshold:
            return 1
        else:
            return 0

    def valid(self):
        all_test_timage =  os.listdir('dataset/valid_input/')
        all_test_image_paths = sorted([ 'dataset/valid_input/' + str(path) for path in all_test_timage])
        all_test_label =  os.listdir('dataset/valid_output/')
        all_test_label_paths = sorted([ 'dataset/valid_output/' + str(path) for path in all_test_label])
        
        rr = 0
        rn = 0
        nr = 0
        nn = 0
        
        for idx in range (len(all_test_image_paths)):
            im = np.array(Image.open(all_test_image_paths[idx]))
            label = np.array(Image.open(all_test_label_paths[idx])) // 200

            pred = self.test([im], [idx])

            rr += np.sum((pred[0] == 1) * (label == 1))
            rn += np.sum((pred[0] == 1) * (label == 0))
            nr += np.sum((pred[0] == 0) * (label == 1))
            nn += np.sum((pred[0] == 0) * (label == 0))

        mIoU = ( 0.5 * rr / (rr + rn + nr) ) + ( 0.5 * nn / (nn + rn + nr) )
        print('[INFO] Validation mIoU')
        print (mIoU)


    def submit(self, test_input, test_input_ids, submission_filename):
        print (len(test_input))
        test_output = self.test(test_input, test_input_ids)

        with open(submission_filename, 'w') as f:
            f.write('id,prediction\n')
            patch_size = 16
            for idx in tqdm(range(len(test_input)), "Writing to CSV"):
                im = test_output[idx]
                print(im.shape)
                img_number = test_input_ids[idx]
                for j in range(0, im.shape[1], patch_size):
                    for i in range(0, im.shape[0], patch_size):
                        patch = im[i:i + patch_size, j:j + patch_size]
                        label = self.patch_to_label(patch)
                        f.writelines("{:03d}_{}_{},{}\n".format(img_number, j, i, label))


    def load_histogram_from_cache(self):
        # Try to load the dataset...
        try:
            with open(self.cache_filename, 'rb') as f:
                data = pickle.load(f)

            return data

        except:
            raise ValueError("[Error] Errors occur in loading data")


    def create_histogram_cache(self):
        # check directory
        if not os.path.exists(self.cache_dirname):
            os.makedirs(self.cache_dirname)
        data = {
            "hist0": self.hist0,
            "hist1": self.hist1
        }
        with open(self.cache_filename, 'wb') as f:
            pickle.dump(data, f)

        print("[Info] Successfully cached histogram!!")



    def __get_color_histogram(self, train_input, train_output):
        """
	    Compute a color histograms based on selected points from all training data

	    :return hist: color histogram
	    """

        hist_res = self.hist_res
        label_thres = self.label_thres


        road_his = np.zeros([hist_res, hist_res, hist_res], dtype=np.float32)
        non_road_his = np.zeros([hist_res, hist_res, hist_res], dtype=np.float32)


        for img, label in zip(train_input, train_output):

            for y in range(0, img.shape[0]):
                for x in range(0, img.shape[1]):

                    pixel = img[y,x,:].astype(np.float32)
                    hist = np.floor(pixel* hist_res / 256.0).astype(np.int32)

                    mean_gt = np.mean(label[y,x]) / 255.0

                    if mean_gt > label_thres:
                        road_his[hist[0], hist[1], hist[2]] = road_his[hist[0], hist[1], hist[2]] + 1
                    else:
                        non_road_his[hist[0], hist[1], hist[2]] = non_road_his[hist[0], hist[1], hist[2]] + 1
            

        road_his = ndimage.gaussian_filter(road_his, sigma=7)
        non_road_his = ndimage.gaussian_filter(non_road_his, sigma=7)


        road_his = road_his / np.sum(road_his)
        non_road_his = non_road_his / np.sum(non_road_his)

        return non_road_his, road_his

    def __get_unaries(self, test_image, road_hist, non_road_his):
        """

        :return: unaries : Nx2 numpy array containing the unary cost for every pixels in I (N = number of pixels in I)
        """

        hist_res = self.hist_res
        lam = self.lam

        H = len (range(0, test_image.shape[0]))
        W = len (range(0, test_image.shape[1]))

        unaries = np.zeros([H, W, 2], dtype = np.float32)
        
        for y in range(0, H):
            for x in range(0, W):

                pixel = test_image[y,x,:].astype(np.float32)

                hist = np.floor(pixel * hist_res / 256).astype(np.int32)

                qf = - np.log(road_hist[hist[0], hist[1], hist[2]] + 1e-9)
                qb = - np.log(non_road_his[hist[0], hist[1], hist[2]] + 1e-9)

                unaries[y, x, 0] = lam * qf / (qf + qb)
                unaries[y, x, 1] = lam * qb / (qf + qb)

        unaries = np.reshape(unaries, [-1,2])

        return unaries

    def __get_pairwise(self, test_image):
        """
        Get pairwise terms for each pairs of pixels on image
        :param image: color image as a numpy array
        :return: pairwise : sparse square matrix containing the pairwise costs for image
        """
        H = len (range(0, test_image.shape[0]))
        W = len (range(0, test_image.shape[1]))
        sigma = 10
        alpha = 1
        hist_res = self.hist_res


        img = test_image.astype(np.float32)

        hist = np.floor(img * hist_res / 256).astype(np.int32)

        row = []
        col = []
        data = []

        for y in range(0, H):
            for x in range(0, W):
        
                if( y+1 < H):
                    row.append( np.ravel_multi_index((y, x), dims=(H, W)) )
                    col.append( np.ravel_multi_index((y+1, x), dims=(H, W)) )
                    B =  np.exp( - np.sum( (hist[y,x,:] - hist[y+1, x,:])**2) / (2 * sigma**2) )
                    data.append(B)

                if(x+1 < W and y-1 >= 0):
                    row.append( np.ravel_multi_index((y, x), dims=(H, W)) )
                    col.append( np.ravel_multi_index((y-1, x+1), dims=(H, W)) )
                    B =  np.exp( - np.sum( (hist[y,x,:] - hist[y-1, x+1,:])**2) / (2 * sigma**2) ) / np.sqrt(2)
                    data.append(B)

                if( x+1 < W):
                    row.append( np.ravel_multi_index((y, x), dims=(H, W)) )
                    col.append( np.ravel_multi_index((y, x+1), dims=(H, W)) )
                    B =  np.exp( - np.sum( (hist[y,x,:] - hist[y, x+1,:])**2) / (2 * sigma**2) )
                    data.append(B)

                if(x+1 < W and y+1 <H ):
                    row.append( np.ravel_multi_index((y, x), dims=(H, W)) )
                    col.append( np.ravel_multi_index((y+1, x+1), dims=(H, W)) )
                    B =  np.exp( - np.sum( (hist[y,x,:] - hist[y+1, x+1,:])**2) / (2 * sigma**2) ) / np.sqrt(2)
                    data.append(B)

        return coo_matrix((data, (row, col)), shape=(H*W, H*W))

