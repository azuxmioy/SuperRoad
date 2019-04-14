from util import *
from dataset import *
from example_method import *
from csv2img import *

if __name__ == '__main__':
    # Define directories and CSV filename
    train_input_dirname = "dataset/train_input/"
    train_output_dirname = "dataset/train_output/"
    test_input_dirname = "dataset/test_input/"
    submission_filename = "example_submission.csv"

    # Load data    
    data = Dataset(train_input_dirname, train_output_dirname, test_input_dirname)

    # Preprocess (Optional)
    data.preprocess()

    # Create method class. May use multiple methods.
    method = ExampleMethod()

    # Train
    method.train(data.train_input, data.train_output)

    # Test (Optional)
    test_output = method.test(data.test_input)

    # Submit: Runs "test", then store output into CSV
    method.submit(data.test_input, data.test_input_ids, submission_filename)
    
    # To visualize test output
    csv_viewer = CSV2Image(submission_filename)
    example_im = csv_viewer.reconstruct_from_labels(7)
    plt.imshow(example_im, cmap = 'gray')
    plt.show()

