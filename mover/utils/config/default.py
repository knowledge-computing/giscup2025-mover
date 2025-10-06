from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.MASK_TYPE = ["continuous", "others"]
_C.DATA.ADD_LOC_PROFILE = False
_C.DATA.ADD_USER_PROFILE = False
_C.DATA.ADD_CITY_EMB = False
_C.DATA.CITY_EMB_SIZE = 4
_C.DATA.TAR_CLUSTER = -1
_C.DATA.TAR_DOW = 0

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.ADD_MOE_MODULE = False
_C.MODEL.HEAD = "ffn1d"
_C.MODEL.PRETRAIN_WEIGHT = ""

_C.MODEL.MOBERT = CN()
_C.MODEL.MOBERT.DROPOUT = 0.1
_C.MODEL.MOBERT.ENCODER = CN()
_C.MODEL.MOBERT.ENCODER.NUM_LAYERS = 6
_C.MODEL.MOBERT.ENCODER.NUM_HEADS = 8
_C.MODEL.MOBERT.ENCODER.EMB_SIZE = 256
_C.MODEL.MOBERT.ENCODER.MAX_SEQ_LEN = 1023
_C.MODEL.MOBERT.ENCODER.PRETRAIN_WEIGHT = ""

_C.MODEL.MOBERT.DECODER = CN()
_C.MODEL.MOBERT.DECODER.NUM_LAYERS = 4
_C.MODEL.MOBERT.DECODER.NUM_HEADS = 8
_C.MODEL.MOBERT.DECODER.EMB_SIZE = 256

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.EPOCHS = 100
_C.TRAIN.WARMUP_EPOCHS = 10
_C.TRAIN.BASE_LR = 1e-4
_C.TRAIN.WARMUP_LR = 1e-7
_C.TRAIN.MIN_LR = 1e-6
_C.TRAIN.WEIGHT_DECAY = 0.01
_C.TRAIN.BASE_LOSS = 'ce'
_C.TRAIN.LOSSES = ['ce']
_C.TRAIN.NUM_WORKERS = 8
_C.TRAIN.PIN_MEM = True
_C.TRAIN.LAYER_DECAY = 1.0
_C.TRAIN.CLIP_GRAD = 1.0
_C.TRAIN.NUM_TRAIN_PER_BATCH = 0
_C.TRAIN.NUM_VAL = 3000
_C.TRAIN.PATIENCE = 10

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 0.9

# Scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.TRAIN_CITIES = []
_C.TEST_CITIES = []
_C.OUTPUT = "./_runs"
_C.EXP_DIR = ""
_C.EXP_NAME = ""
_C.MODEL_NAME = None

_C.CHECKPOINT_PATH = None
_C.LOG_PATH = None
_C.SAVE_CONFIG_FILE = None
_C.SAVE_EVERY_EPOCH = 20
_C.EVAL_EVERY_EPOCH = 1

_C.USE_WANDB = False
_C.SEED = 1
_C.VERSION = 0
