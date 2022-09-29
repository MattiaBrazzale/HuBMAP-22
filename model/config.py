SEED = 42
IMG_SIZE = 768
BATCH_SIZE = 1
ACCUMULATION = 24
EPOCHS = 200
NUM_WORKERS = 4
MAX_LEARNING_RATE = 1e-3
ENCODER_LEARNING_RATE = 5e-5
DECODER_LEARNING_RATE = 5e-5

DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_PATH = 'hubmap-organ-segmentation/train_images/'
TEST_PATH = 'hubmap-organ-segmentation/test_images/'

VAL_FOLD = 0
NUM_FOLD = 5

TRAIN = False
SUBMIT = True

WEIGHTS_PATH = '' #path for the saved weights to use for the submission
