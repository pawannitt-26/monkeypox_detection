import os

# Directory settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Data settings
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
IMAGE_SIZE = (224, 224)
IMG_HEIGHT = 224
IMG_WIDTH = 224
LEARNING_RATE = 0.003
SEED = 42

# Augmentation parameters
AUGMENTATION_PARAMS = {
    "rotation_range": 45,
    "width_shift_range": 0.3,
    "height_shift_range": 0.3,
    "shear_range": 45,
    "zoom_range": [0.8, 1.25],
    "horizontal_flip": True,
    "vertical_flip": True,
    "brightness_range": [0.1, 2],
    "fill_mode": 'constant',
}

# Model configuration
MODEL_CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'monkeypox_detection_model.h5')
