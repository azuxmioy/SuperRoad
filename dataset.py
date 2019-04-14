from util import *
import ipdb

class Dataset:
    def __init__(self, train_input_dirname, train_output_dirname, test_input_dirname):
        # Initialize dataset root and define cache path
        if train_input_dirname.split("/")[-1] == "":
            self.dataset_root = "/".join(train_input_dirname.split("/")[:-2])
        else:
            self.dataset_root = "/".join(train_input_dirname.split("/")[:-1])
        self.cache_dirname = os.path.join(self.dataset_root, "cache")
        self.cache_filename = os.path.join(self.cache_dirname, "dataset_cache.pkl")

        # Check if cached dataset exists
        if os.path.exists(self.cache_filename):
            print("[Info]: Found dataset cache. Load from cache!")
            data = self.load_filename_dataset_from_cache()
            self.train_input = data["train_input"]
            self.train_output = data["train_output"]
            self.test_input = data["test_input"]
            self.test_input_ids = data["test_input_ids"]

        else:
            # Load raw data from disk
            self.train_input = []
            self.train_output = []
            self.test_input = []
            self.test_input_ids = []
            print("[Info]: Can't find dataset cache.")
            print("[Info]: Loading raw data from disk.")
            for filename in tqdm(os.listdir(train_input_dirname), "Loading training input"):
                self.train_input.append(np.array(Image.open(train_input_dirname + filename)))
            for filename in tqdm(os.listdir(train_output_dirname), "Loading training output"):
                self.train_output.append(np.array(Image.open(train_output_dirname + filename)))
            for filename in tqdm(os.listdir(test_input_dirname), "Loading testing input"):
                self.test_input.append(np.array(Image.open(test_input_dirname + filename)))
                img_number = int(re.search(r"\d+", filename).group(0))
                self.test_input_ids.append(img_number)

            # Create dataset cache from raw data
            self.create_dataset_cache()

    def __kmeans(self, X, k):
        # Initialize centers as k randomly chosen points
        c = X[np.random.choice(X.shape[0], size=k, replace=False)]
    
        z_old = None
        converged = False
        iteration = 0
        while not converged:
            # Compute assignments
            distances = np.sum((X[:, np.newaxis] - c[np.newaxis, :])**2, axis=2) # Squared
            z = np.argmin(distances, axis=1)
    
            # Compute mean for each cluster
            valid_clusters = []
            for i in range(c.shape[0]):
                points = X[z == i]
                if len(points) > 0:
                    c[i] = np.mean(points, axis=0)
                    valid_clusters.append(True)
                else:
                    valid_clusters.append(False)
            
            # Drop empty clusters
            c = c[valid_clusters]
            
            if z_old is not None and (z == z_old).all():
                converged = True
            z_old = z
            iteration += 1
        return z, c
  
    def __quantize(self, im, k):
        X = im.reshape(-1, 3).astype('float32')/255
        z, c = self.__kmeans(X, k)
        reconstruction = c[z].reshape(im.shape)
        return reconstruction
  
    def preprocess(self, num_buckets = 4):
        for i in tqdm(range(len(self.train_input)), "Preprocessing training input"):
            im = self.train_input[i]
            self.train_input[i] = self.__quantize(im, num_buckets)
        for i in tqdm(range(len(self.test_input)), "Preprocessing testing input"):
            im = self.test_input[i]
            self.test_input[i] = self.__quantize(im, num_buckets)

    # Load data from pkl cache
    def load_filename_dataset_from_cache(self):
        # Try to load the dataset...
        try:
            with open(self.cache_filename, 'rb') as f:
                data = pickle.load(f)

            return data

        except:
            raise ValueError("[Error] Errors occur in loading data")

    # Create dataset cache from data
    def create_dataset_cache(self):
        # check directory
        if not os.path.exists(self.cache_dirname):
            os.makedirs(self.cache_dirname)
        data = {
            "train_input": self.train_input,
            "train_output": self.train_output,
            "test_input": self.test_input,
            "test_input_ids": self.test_input_ids
        }
        with open(self.cache_filename, 'wb') as f:
            pickle.dump(data, f)

        print("[Info] Successfully cached filename dataset!!")


