from importlib import util
import sys

spec = util.spec_from_file_location("config", sys.argv[1])
config = util.module_from_spec(spec)
spec.loader.exec_module(config)
sys.stdout.write(config.DATASET)