SEED = 42 
IMG_SIZE = 768 
BATCH_SIZE = 1
ACCUMULATION = 24 # for how many iterations the gradient is accumulated before doing an optimization step
EPOCHS = 200
NUM_WORKERS = 4
MAX_LEARNING_RATE = 1e-3
ENCODER_LEARNING_RATE = 5e-5
DECODER_LEARNING_RATE = 5e-5

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_PATH = 'hubmap-organ-segmentation/train_images/'  # path for the train images
TEST_PATH = 'hubmap-organ-segmentation/test_images/'  # path for the test images

VAL_FOLD = 0  # which fold is used as validation set
NUM_FOLD = 5  # number of folds in which the train set is split

TRAIN = False   # if True, training is performed
SUBMIT = True   # if True, inference on the test set is performed

WEIGHTS_PATH = '' # path for the saved weights to use for the submission
TTA = True    # if True, Test Time Augmentations are performed
