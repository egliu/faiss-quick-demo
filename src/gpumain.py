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
    index_list = []
    begin_time = time.time()
    if config['index_type'] == IndexType.Flat.name:
        print(IndexType.Flat.name)
        import gpufaiss.flatgpu
        index_list = gpufaiss.flatgpu.FlatGpu(config)
    elif config['index_type'] == IndexType.IVFFlat.name:
        print(IndexType.IVFFlat.name)
        import gpufaiss.ivfflatgpu
        index_list = gpufaiss.ivfflatgpu.IVFFlatGpu(config)
    elif config['index_type'] == IndexType.IVFPQ.name:
        print(IndexType.IVFPQ.name)
        import gpufaiss.ivfpqgpu
        index_list = gpufaiss.ivfpqgpu.IVFPQGpu(config)
    else:
        print("type ", config['index_type'], " is not supported, exit")
    print("duration : ", time.time()-begin_time, " s")
    time.sleep(2000)
    print("End : %s" % time.ctime())
