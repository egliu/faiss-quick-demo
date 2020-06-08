import json
from enum import Enum
import time


def load_config():
    with open("/home/faiss-quick-demo/config/config.json", encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config


if __name__ == '__main__':
    config = load_config()
    IndexType = Enum('IndexType', ('Flat', 'IVFFlat', 'IVFPQ'))
    if config['index_type'] == IndexType.Flat.name:
        print(IndexType.Flat.name)
        import flatgpu
        index = flatgpu.FlatGpu(config)
    elif config['index_type'] == IndexType.IVFFlat.name:
        print(IndexType.IVFFlat.name)
    elif config['index_type'] == IndexType.IVFPQ.name:
        print(IndexType.IVFPQ.name)
    else:
        print("type ", config['index_type'], " is not supported, exit")
    print("Start : %s" % time.ctime())
    time.sleep(30)
    print("End : %s" % time.ctime())
