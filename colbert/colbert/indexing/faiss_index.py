import sys
import time
import math
import faiss
import torch

import numpy as np

from colbert.indexing.faiss_index_gpu import FaissIndexGPU
from colbert.utils.utils import print_message


class FaissIndex():
    """
    Train a IVFPQ quantizer to speed up indexing, improving UX.
    """
    def __init__(self, dim, partitions):
        self.dim = dim
        self.partitions = partitions

        self.gpu = FaissIndexGPU()
        self.quantizer, self.index = self._create_index()
        self.offset = 0

    def _create_index(self):
        quantizer = faiss.IndexFlatL2(self.dim)  # faiss.IndexHNSWFlat(dim, 32) # hierarchical navigable small world graph only cpu
        index = faiss.IndexIVFPQ(quantizer, self.dim, self.partitions, 16, 8)
        return quantizer, index

    def train(self, train_data):
        print_message(f"#> Training now (using {self.gpu.ngpu} GPUs)...")

        if self.gpu.ngpu > 0:
            self.gpu.training_initialize(self.index, self.quantizer)

        s = time.time()
        self.index.train(train_data)
        print(time.time() - s)

        if self.gpu.ngpu > 0:
            self.gpu.training_finalize()

    def add(self, data):
        print_message(f"Add data with shape {data.shape} (offset = {self.offset})..")

        if self.gpu.ngpu > 0 and self.offset == 0:
            self.gpu.adding_initialize(self.index)

        if self.gpu.ngpu > 0:
            self.gpu.add(self.index, data, self.offset)
        else:
            self.index.add(data)

        self.offset += data.shape[0]

    def save(self, output_path):
        print_message(f"Writing index to {output_path} ...")

        self.index.nprobe = 10  # just a default
        faiss.write_index(self.index, output_path)


class FaissFlatIndex():
    """
    Use flat indexer to ensure reproductivity
    """
    def __init__(self, dim, partitions, index_ttype='ivf'):
        self.dim = dim
        self.partitions = partitions
        self.index_ttype = index_ttype
        self.gpu = FaissIndexGPU()
        self.quantizer, self.index = self._create_index()
        self.offset = 0

    def _create_index(self):
        config = faiss.GpuIndexFlatConfig()  # config for gpu
        config.device = 0
        if self.index_ttype == 'ivf':
            quantizer = faiss.IndexFlatL2(self.dim)
            index = faiss.GpuIndexIVFFlat(self.gpu.gpu_resources[0], self.dim, self.partitions, faiss.METRIC_L2)
            return quantizer, index
        elif self.index_ttype == 'brute-forece':
            index = faiss.GpuIndexFlatIP(self.gpu.gpu_resources[0], self.dim, config)   # flat index no inv indexing
            return None, index
        else:
            raise NotImplementedError

    def add(self, data):
        print_message(f"Add data with shape {data.shape} (offset = {self.offset})..")

        if self.gpu.ngpu > 0 and self.offset == 0:
            self.index.add(data)    # add embs to flat index
        else:
            self.index.add_with_ids(data, np.arange(self.offset, self.offset+len(data)))

        self.offset += data.shape[0]

    def train(self, train_data):
        print_message(f"#> Training now (using {self.gpu.ngpu} GPUs)...")

        if self.gpu.ngpu > 0:
            self.gpu.training_initialize(self.index, self.quantizer)

        s = time.time()
        self.index.train(train_data)
        print(time.time() - s)

        if self.gpu.ngpu > 0:
            self.gpu.training_finalize()

    def save(self, output_path):
        print_message(f"Writing index to {output_path} ...")

        self.index.nprobe = 10  # just a default
        self.index = faiss.index_gpu_to_cpu(self.index)     # transfer the index to cpu
        faiss.write_index(self.index, output_path)
