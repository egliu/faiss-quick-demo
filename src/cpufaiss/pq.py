import faiss                     # make faiss available
import numpy as np
import time


def PQCpu(config):
    print("PQCpu, ", config)
    d = config['dimension']                     # dimension
    nb = config['db_size']                      # database size
    nq = config['query_num']                    # nb of queries
    topk = config['top_k']
    m = config['sub_quantizers']
    nbits = config['bits_per_code']
    search_repeat = 10

    index_list = []
    create_ave_duration = 0
    search_ave_duration = 0

    for i in range(config['db_num']):
        np.random.seed(i)                        # make reproducible
        xb = np.random.random((nb, d)).astype('float32')
        xb[:, 0] += np.arange(nb) / 1000.
        begin_time = time.time()

        index_pq = faiss.IndexPQ(d, m, nbits)
        assert not index_pq.is_trained
        index_pq.train(xb)
        assert index_pq.is_trained

        index_pq.add(xb)                  # add vectors to the index
        duration = time.time()-begin_time
        create_ave_duration += duration
        index_list.append(index_pq)
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
