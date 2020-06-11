import faiss                     # make faiss available
import numpy as np
import time


def FlatGpu(config):
    print("FlatGpu, ", config)
    d = config['dimension']                     # dimension
    nb = config['db_size']                      # database size
    nq = config['query_num']                    # nb of queries
    topk = config['top_k']
    search_repeat = 10

    xq = np.random.random((nq, d)).astype('float32')
    xq[:, 0] += np.arange(nq) / 1000.

    res = faiss.StandardGpuResources()  # use a single GPU
    # temp memory
    if config["temp_memory"] == 0:
        res.noTempMemory()
    elif config["temp_memory"] != -1:
        res.setTempMemory(config["temp_memory"])

    index_list = []
    create_ave_duration = 0
    search_ave_duration = 0

    if config['test_batch_write'] == True:
        batch_write_ave_duration = 0
        batch_write_num = config['write_batch_num']
        batch_write_time = int(nb/config['write_batch_num'])
        print("batch_write_time = ", batch_write_num)
        for i in range(config['db_num']):
            index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index
            # make it a flat GPU index
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
            batch_write_ave_one_lib = 0
            for j in range(batch_write_time):
                np.random.seed(i*batch_write_time+j)
                xb = np.random.random((batch_write_num, d)).astype('float32')
                xb[:, 0] += np.arange(batch_write_num) / 1000.
                begin_time = time.time()
                gpu_index_flat.add(xb)
                duration = time.time()-begin_time
                batch_write_ave_one_lib += duration
                batch_write_ave_duration += duration
            print("batch_write_ave_one_lib = ",
                  (batch_write_ave_one_lib/batch_write_time)*1000*1000, " us")
            index_list.append(gpu_index_flat)
        print("batch_write_ave_duration = ", (batch_write_ave_duration /
                                              len(index_list)/batch_write_time)*1000*1000, " us")

        return index_list

    # Using a flat index
    for i in range(config['db_num']):
        np.random.seed(i)                        # make reproducible
        xb = np.random.random((nb, d)).astype('float32')
        xb[:, 0] += np.arange(nb) / 1000.
        begin_time = time.time()
        index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index
        # make it a flat GPU index
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        gpu_index_flat.add(xb)         # add vectors to the index
        duration = time.time()-begin_time
        create_ave_duration += duration
        index_list.append(gpu_index_flat)
        if i == 0:
            gpu_index_flat.search(xb[:5], 4)
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
