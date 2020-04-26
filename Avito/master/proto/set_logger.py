from logging import StreamHandler, FileHandler
from logging import Formatter, INFO, DEBUG, getLogger

FILEPATH = './result_tmp/train.py.log'

def set_logger(FILEPATH):
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(FILEPATH, 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

logger = getLogger(__name__)
logger.setLevel(DEBUG)
set_logger(FILEPATH)

