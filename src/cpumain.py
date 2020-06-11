import json
from enum import Enum
import time


def load_config():
    with open("/home/faiss-quick-demo/config/config-cpu.json", encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config


if __name__ == '__main__':
    config = load_config()
    IndexType = Enum('IndexType', ('Flat', 'IVFFlat', 'PQ', 'IVFPQ'))
    index_list = []
    begin_time = time.time()
    if config['index_type'] == IndexType.Flat.name:
        print(IndexType.Flat.name)
        import cpufaiss.flat
        index_list = cpufaiss.flat.FlatCpu(config)
    elif config['index_type'] == IndexType.IVFFlat.name:
        print(IndexType.IVFFlat.name)
        import cpufaiss.ivfflat
        index_list = cpufaiss.ivfflat.IVFFlatCpu(config)
    elif config['index_type'] == IndexType.PQ.name:
        print(IndexType.PQ.name)
        import cpufaiss.pq
        index_list = cpufaiss.pq.PQCpu(config)
    elif config['index_type'] == IndexType.IVFPQ.name:
        print(IndexType.IVFPQ.name)
        import cpufaiss.ivfpq
        index_list = cpufaiss.ivfpq.IVFPQCpu(config)
    else:
        print("type ", config['index_type'], " is not supported, exit")
    print("duration : ", time.time()-begin_time, " s")
    time.sleep(2000)
    print("End : %s" % time.ctime())
