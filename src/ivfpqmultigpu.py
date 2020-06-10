import faiss                     # make faiss available
import numpy as np
import time


def IVFPQMultiGpu(config):
    print("IVFPQMultiGpu, ", config)
    d = config['dimension']                     # dimension
    nb = config['db_size']                      # database size
    nq = config['query_num']                    # nb of queries
    k = config['top_k']
    config_gpus = config['gpus']

    ngpus = faiss.get_num_gpus()
    print("number of GPUs:", ngpus, ",running on gpus:", config_gpus)
    gpus = range(config_gpus)
    res = [faiss.StandardGpuResources() for _ in gpus]
    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    for i, res in zip(gpus, res):
        vdev.push_back(i)
        vres.push_back(res)

    index_list = []

    for i in range(config['db_num']):
        # Using an IVFPQ index
        np.random.seed(i)
        xb = np.random.random((nb, d)).astype('float32')
        xb[:, 0] += np.arange(nb) / 1000.
        nlist = config['nlist']
        m = config['sub_quantizers']
        code = config['bits_per_code']
        # begin_time = time.time()
        quantizer = faiss.IndexFlatL2(d)  # the other index
        index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, code)
        # here we specify METRIC_L2, by default it performs inner-product search

        # build the index
        gpu_index_ivfpq = faiss.index_cpu_to_gpu_multiple(
            vres, vdev, index_ivfpq)
        gpu_index_ivfpq.referenced_objects = res

        assert not gpu_index_ivfpq.is_trained
        gpu_index_ivfpq.train(xb)        # add vectors to the index
        assert gpu_index_ivfpq.is_trained

        gpu_index_ivfpq.add(xb)          # add vectors to the index
        print(i, ",size = ", gpu_index_ivfpq.ntotal)
        index_list.append(gpu_index_ivfpq)
    return index_list
