import configparser

class ConfigContainer():

    def __init__(self):
        config = configparser.RawConfigParser()
        config.read('config.properties')
        self.estimatorConfigDict = dict(config.items('ESTIMATOR_SECTION'))




