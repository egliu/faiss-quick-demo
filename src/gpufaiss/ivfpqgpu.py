import faiss                     # make faiss available
import numpy as np
import time


def IVFPQGpu(config):
    if config['multi_gpu'] == True:
        import ivfpqmultigpu
        return ivfpqmultigpu.IVFPQMultiGpu(config)
    print("IVFPQGpu, ", config)
    d = config['dimension']                     # dimension
    nb = config['db_size']                      # database size
    nq = config['query_num']                    # nb of queries
    topk = config['top_k']
    m = config['sub_quantizers']
    nbits = config['bits_per_code']
    search_repeat = 10
    nlist = config['nlist']
    nprobe = config['nprobe']

    res = faiss.StandardGpuResources()  # use a single GPU
    # temp memory
    if config["temp_memory"] == 0:
        print("set noTempMemory")
        res.noTempMemory()
    elif config["temp_memory"] != -1:
        print("set temp_memory to ", config["temp_memory"])
        res.setTempMemory(config["temp_memory"]*1024*1024)

    index_list = []
    create_ave_duration = 0
    search_ave_duration = 0

    for i in range(config['db_num']):
        np.random.seed(i)
        xb = np.random.random((nb, d)).astype('float32')
        xb[:, 0] += np.arange(nb) / 1000.
        begin_time = time.time()
        quantizer = faiss.IndexFlatL2(d)  # the other index
        index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        gpu_index_ivfpq = faiss.index_cpu_to_gpu(res, 0, index_ivfpq)

        assert not gpu_index_ivfpq.is_trained
        gpu_index_ivfpq.train(xb)        # add vectors to the index
        assert gpu_index_ivfpq.is_trained

        gpu_index_ivfpq.add(xb)          # add vectors to the index
        gpu_index_ivfpq.nprobe = nprobe
        duration = time.time()-begin_time
        create_ave_duration += duration
        index_list.append(gpu_index_ivfpq)
        if i == 0:
            gpu_index_ivfpq.search(xb[:5], 4)
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
