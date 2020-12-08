import luigi
import gokart
import dajare_detector

if __name__ == '__main__':
    luigi.configuration.LuigiConfigParser.add_config_path('./conf/param.ini')
    gokart.run()
