from configparser import ConfigParser

def get_all_configs(cfg_file):
    config = ConfigParser()
    config.read(cfg_file)

    return config