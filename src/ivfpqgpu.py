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
    k = config['top_k']
    nlist = config['nlist']
    m = config['sub_quantizers']
    code = config['bits_per_code']

    res = faiss.StandardGpuResources()  # use a single GPU
    # temp memory
    if config["temp_memory"] == 0:
        print("set noTempMemory")
        res.noTempMemory()
    elif config["temp_memory"] != -1:
        print("set temp_memory to ", config["temp_memory"])
        res.setTempMemory(config["temp_memory"]*1024*1024)

    index_list = []
    ave_duration = 0
    data_prepare_duration = 0
    data_train_add_duration = 0

    for i in range(config['db_num']):
        # Using an IVFPQ index
        begin_time = time.time()
        np.random.seed(i)
        xb = np.random.random((nb, d)).astype('float32')
        xb[:, 0] += np.arange(nb) / 1000.
        duration = time.time()-begin_time
        print(i, ", data prepare duration is ", duration)
        if i > 0:
            data_prepare_duration += duration
        begin_time = time.time()
        quantizer = faiss.IndexFlatL2(d)  # the other index
        index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, code)
        # here we specify METRIC_L2, by default it performs inner-product search

        # make it an IVFPQ GPU index
        gpu_index_ivfpq = faiss.index_cpu_to_gpu(res, 0, index_ivfpq)

        assert not gpu_index_ivfpq.is_trained
        gpu_index_ivfpq.train(xb)        # add vectors to the index
        assert gpu_index_ivfpq.is_trained

        gpu_index_ivfpq.add(xb)          # add vectors to the index
        duration = time.time()-begin_time
        print(i, ", data train add duration is ", duration)
        if i > 0:
            data_train_add_duration += duration
        print(i, ",size = ", gpu_index_ivfpq.ntotal)
        # duration = time.time()-begin_time
        # print(i, ", duration = ", duration, " s")
        # D, I = gpu_index_ivfpq.search(xb[:5], 4)
        # print(I)
        # print(D)
        # ave_duration += duration
        index_list.append(gpu_index_ivfpq)
    print("data_prepare_duration = ", data_prepare_duration,
          ",data_train_add_duration = ", data_train_add_duration)

    # print("begin search, index_list len = ", len(index_list))
    # print("construct index aver time = ", ave_duration/len(index_list), " s")
    # ave_duration = 0

    # for i in range(config['db_num']):
    #     np.random.seed(i+config['db_num'])
    #     xq = np.random.random((nq, d)).astype('float32')
    #     xq[:, 0] += np.arange(nq) / 1000.
    #     begin_time = time.time()
    #     D, I = index_list[i].search(xq, k)  # actual search
    #     duration = time.time()-begin_time
    #     print(i, ", duration = ", duration, " s")
    #     # print(I[:5])                   # neighbors of the 5 first queries
    #     # print(I[-5:])                  # neighbors of the 5 last queries
    #     ave_duration += duration

    # print("search index aver time = ", ave_duration/len(index_list), " s")
    return index_list
