from util import *
from method import *

class ExampleMethod(Method):
    def train(self, train_input, train_output):
        pass

    def test(self, test_input):
        test_output = []
        for i in tqdm(range(len(test_input)), "Running test"):
            im = test_input[i]
            h, w, _ = im.shape
            prediction = np.zeros((h, w))
            test_output.append(prediction)
        return test_output

    # May want to consider smarter ways than just taking threshold
    def patch_to_label(self, patch):
        # percentage of pixels > 1 required to assign a foreground label to a patch
        foreground_threshold = 0.25
        df = np.mean(patch)
        if df > foreground_threshold:
            return 1
        else:
            return 0

    def submit(self, test_input, test_input_ids, submission_filename):
        test_output = self.test(test_input)
        with open(submission_filename, 'w') as f:
            f.write('id,prediction\n')
            patch_size = 16
            for idx in tqdm(range(len(test_input)), "Writing to CSV"):
                im = test_output[idx]
                img_number = test_input_ids[idx]
                for j in range(0, im.shape[1], patch_size):
                    for i in range(0, im.shape[0], patch_size):
                        patch = im[i:i + patch_size, j:j + patch_size]
                        label = self.patch_to_label(patch)
                    f.writelines("{:03d}_{}_{},{}\n".format(img_number, j, i, label))

