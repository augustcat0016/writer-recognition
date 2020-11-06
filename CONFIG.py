DATA_PATH = './dataset'
CACHE_DIR = './bert-base-chinese-model'
FILENAME_CONFIG = 'config.json'
FILENAME_VOCAB = 'vocab.txt'
FILENAME_MODEL = 'pytorch_model.bin'
MODEL_SAVE_DIR = './model'

LABEL_LIST = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}
WRITER_LIST = [item[0] for item in LABEL_LIST.items()]
LABEL_NUM = 5

LIMIT_SIZE = 158
SEQ_LEN = 160
VALID_RATIO = 0.2

NUM_EPOCH = 50
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
EPSILON = 1e-8