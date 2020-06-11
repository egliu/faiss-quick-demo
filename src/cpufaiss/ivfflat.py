import faiss                     # make faiss available
import numpy as np
import time


def IVFFlatCpu(config):
    print("IVFFlatCpu, ", config)
    d = config['dimension']                     # dimension
    nb = config['db_size']                      # database size
    nq = config['query_num']                    # nb of queries
    topk = config['top_k']
    nlist = config['nlist']
    nprobe = config['nprobe']
    search_repeat = 10

    index_list = []
    create_ave_duration = 0
    search_ave_duration = 0

    for i in range(config['db_num']):
        np.random.seed(i)                        # make reproducible
        xb = np.random.random((nb, d)).astype('float32')
        xb[:, 0] += np.arange(nb) / 1000.
        begin_time = time.time()
        # Using an IVF index
        quantizer = faiss.IndexFlatL2(d)  # the other index
        index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        # here we specify METRIC_L2, by default it performs inner-product search

        assert not index_ivf.is_trained
        index_ivf.train(xb)
        assert index_ivf.is_trained

        index_ivf.add(xb)          # add vectors to the index
        index_ivf.nprobe = nprobe
        duration = time.time()-begin_time
        create_ave_duration += duration
        index_list.append(index_ivf)
    print("craete ave duration = ", create_ave_duration/len(index_list), " s")
    if len(index_list) == 0:
        return index_list
    for i in range(len(index_list)):
        for j in range(search_repeat):
            np.random.seed(i*search_repeat+j+config['db_num'])
            xq = np.random.random((nq, d)).astype('float32')
            xq[:, 0] += np.arange(nq) / 1000.
            begin_time = time.time()
            index_list[i].search(xq, topk)  # actual search
            duration = time.time()-begin_time
            search_ave_duration += duration

    print("search index aver time = ", search_ave_duration /
          len(index_list)/search_repeat, " s")
    return index_list
