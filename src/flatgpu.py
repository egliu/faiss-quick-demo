import faiss                     # make faiss available
import numpy as np
import time


def FlatGpu(config):
    print("FlatGpu, ", config)
    d = config['dimension']                     # dimension
    nb = config['db_size']                      # database size
    nq = config['query_num']                    # nb of queries
    np.random.seed(1234)                        # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    xb[:, 0] += np.arange(nb) / 1000.
    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.

    res = faiss.StandardGpuResources()  # use a single GPU
    # temp memory
    if config["temp_memory"] == 0:
        res.noTempMemory()
    else:
        res.setTempMemory(config["temp_memory"])

    index_list = []

    # Using a flat index
    for i in range(config['db_num']):
        print(i)
        begin_time = time.process_time()

        index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index

        # make it a flat GPU index
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

        gpu_index_flat.add(xb)         # add vectors to the index
        index_list.append(gpu_index_flat)
        print(gpu_index_flat.ntotal)
        print("one time duration: ", time.process_time()-begin_time)

    return index_list
