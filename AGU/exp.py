import logging
from lib_dataset.data_store import DataStore

class Exp:
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger('exp')
        self.data_store = DataStore(args)

    def load_data(self):
        pass