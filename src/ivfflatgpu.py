import faiss                     # make faiss available
import numpy as np


def IVFFlatGpu(config):
    print("IVFFlatGpu, ", config)
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

    # Using an IVF index

    nlist = config['nlist']
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    # here we specify METRIC_L2, by default it performs inner-product search

    # make it an IVF GPU index
    gpu_index_ivf = faiss.index_cpu_to_gpu(res, 0, index_ivf)

    assert not gpu_index_ivf.is_trained
    gpu_index_ivf.train(xb)        # add vectors to the index
    assert gpu_index_ivf.is_trained

    gpu_index_ivf.add(xb)          # add vectors to the index
    print(gpu_index_ivf.ntotal)
    return gpu_index_ivf
