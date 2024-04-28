import numpy as np
from scipy.stats import dirichlet
from dataclasses import dataclass
import itertools
import random
import torch.nn.functional as F
import multiprocess as mp

def sort_bin(b):
    b_view = np.ascontiguousarray(b).view(np.dtype((np.void, b.dtype.itemsize * b.shape[1])))
    return b[np.argsort(b_view.ravel())] #as per Divakar's suggestion

@dataclass
class DataArgs:
    n_concept: int = 4
    seq_len: int = 128
    nb_mem: int = 16
    piece_rate: float=0.8
    beta: float=1.0
    cluster: bool=False

def sort_bin(b):
    b_view = np.ascontiguousarray(b).view(np.dtype((np.void, b.dtype.itemsize * b.shape[1])))
    return b[np.argsort(b_view.ravel())] #as per Divakar's suggestion

class Dataset:
    def __init__(self, args: DataArgs):
        self.args = args

        # for easy access
        self.n_concept = args.n_concept # number of concepts
        self.seq_len = args.seq_len # seq length
        self.nb_mem = args.nb_mem # number of memory patterns
        self.piece_rate = args.piece_rate
        self.beta = args.beta
        self.vocab_size = 2**self.n_concept + 1 # one extra token to mean unknown token, might be used later
        self.tok_range = list(np.arange(self.vocab_size))
        self.cluster_strategy = args.cluster

        assert self.nb_mem <= 2**self.n_concept


        all_vectors = list(itertools.product([0, 1], repeat=self.n_concept))
        self.mem_patterns = np.array(random.sample(all_vectors, self.nb_mem))
        self.mem_patterns = sort_bin(self.mem_patterns)

        self.pi_prob = {}

        for mem_ind in range(self.nb_mem):
            neigbors = self.neighbors(mem_ind)
            target = self.mem_patterns[mem_ind]

            unnomarlized_prob = np.zeros(self.vocab_size)
            for neighbor in neigbors:
                hamming_distance = np.count_nonzero(target!=neighbor)
                unnomarlized_prob[self.tokenize(neighbor)] = np.exp(-hamming_distance/self.beta)

            prob = unnomarlized_prob/np.sum(unnomarlized_prob)

            self.pi_prob[mem_ind] = prob

    def neighbors(self, target):

        all_vectors = np.array(list(itertools.product([0, 1], repeat=self.n_concept)))
        target_mem = self.mem_patterns[target]

        indices_to_remove = np.where(np.all(all_vectors == target_mem, axis=1))[0].tolist()
        if self.cluster_strategy:
            cluster_id = 1-target_mem[0]
            all_vectors_minus_one = np.array(list(itertools.product([0, 1], repeat=self.n_concept-1)))
            for vector in all_vectors_minus_one:
                vector_to_remove = np.insert(vector, 0, cluster_id)
                indices_to_remove.append(self.tokenize(vector_to_remove))

        indices_to_remove = np.unique(indices_to_remove)

        return np.delete(all_vectors, indices_to_remove, axis=0)

    def tokenize(self, vector):
        if vector[1] == -1:
            return self.vocab_size - 1
        return int(''.join(map(str, vector.astype(int))), 2)

    def generate_piece(self, mem_ind):
        # print(self.pi_prob[mem_ind])
        return np.random.choice(self.tok_range, p=self.pi_prob[mem_ind])

    def gen_seq(self, mem_ind=None):
        if mem_ind == None:
            # select a memory index to use
            mem_ind = np.random.randint(0, self.nb_mem)
        target = self.mem_patterns[mem_ind]

        seq = []
        # probabilities = np.random.dirichlet(np.ones(self.vocab_size-1)/1)
        # probabilities = np.append(probabilities, 0)

        while len(seq) < self.seq_len - 1:
            if random.random() < self.piece_rate:
                next_token = self.generate_piece(mem_ind)
            else:
                flipped_vector = np.random.randint(2, size=self.n_concept)
                next_token = self.tokenize(flipped_vector)
                # next_token = np.random.choice(self.tok_range, p=probabilities)

            seq += [next_token]

        next_token = self.generate_piece(mem_ind)
        seq += [next_token]

        seq += [self.tokenize(target)]

        return seq

    def gen_batch(self, batch_size: int):
        seqs = []
        for _ in range(batch_size):
            seq = self.gen_seq()
            seqs += seq
        x = np.array(seqs).reshape(batch_size, self.seq_len + 1)
        return x

def iterate_batches(dataset,
                    batch_size: int = 20,
                    num_workers: int = 60,
                    seed: int = 42):
    def worker(queue):
        while True:
            x = dataset.gen_batch(batch_size)
            queue.put((x))

    q = mp.Queue(maxsize=1000)
    processes = [mp.Process(target=worker, args=(q,)) for i in range(num_workers)]
    for p in processes:
        p.start()

    seq = []
    outputs_seq = []
    count = 0
    try:
        while True:
            x = q.get()
            yield (x[:,:-1], x[:,1:])
    except:
        for p in processes:
            p.kill()