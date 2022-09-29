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
TRAIN_IMG = 'hubmap-organ-segmentation/train_images/'
TEST_IMG = 'hubmap-organ-segmentation/test_images/'

val_fold = 0
num_fold = 5
