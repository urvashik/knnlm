import argparse
import os
import numpy as np
import faiss
import time


parser = argparse.ArgumentParser()
parser.add_argument('--dstore_mmap', type=str, help='memmap where keys and vals are stored')
parser.add_argument('--dstore_size', type=int, help='number of items saved in the datastore memmap')
parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
parser.add_argument('--dstore_fp16', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=1, help='random seed for sampling the subset of vectors to train the cache')
parser.add_argument('--ncentroids', type=int, default=4096, help='number of centroids faiss should learn')
parser.add_argument('--code_size', type=int, default=64, help='size of quantized vectors')
parser.add_argument('--probe', type=int, default=8, help='number of clusters to query')
parser.add_argument('--faiss_index', type=str, help='file to write the faiss index')
parser.add_argument('--num_keys_to_add_at_a_time', default=1000000, type=int,
                    help='can only load a certain amount of data to memory at a time.')
parser.add_argument('--starting_point', type=int, help='index to start adding keys at')
args = parser.parse_args()

print(args)

if args.dstore_fp16:
    keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int16, mode='r', shape=(args.dstore_size, 1))
else:
    keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float32, mode='r', shape=(args.dstore_size, args.dimension))
    vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))

if not os.path.exists(args.faiss_index+".trained"):
    # Initialize faiss index
    quantizer = faiss.IndexFlatL2(args.dimension)
    index = faiss.IndexIVFPQ(quantizer, args.dimension,
        args.ncentroids, args.code_size, 8)
    index.nprobe = args.probe

    print('Training Index')
    np.random.seed(args.seed)
    random_sample = np.random.choice(np.arange(vals.shape[0]), size=[min(1000000, vals.shape[0])], replace=False)
    start = time.time()
    # Faiss does not handle adding keys in fp16 as of writing this.
    index.train(keys[random_sample].astype(np.float32))
    print('Training took {} s'.format(time.time() - start))

    print('Writing index after training')
    start = time.time()
    faiss.write_index(index, args.faiss_index+".trained")
    print('Writing index took {} s'.format(time.time()-start))

print('Adding Keys')
index = faiss.read_index(args.faiss_index+".trained")
start = args.starting_point
start_time = time.time()
while start < args.dstore_size:
    end = min(args.dstore_size, start+args.num_keys_to_add_at_a_time)
    to_add = keys[start:end].copy()
    index.add_with_ids(to_add.astype(np.float32), np.arange(start, end))
    start += args.num_keys_to_add_at_a_time

    if (start % 1000000) == 0:
        print('Added %d tokens so far' % start)
        print('Writing Index', start)
        faiss.write_index(index, args.faiss_index)

print("Adding total %d keys" % start)
print('Adding took {} s'.format(time.time() - start_time))
print('Writing Index')
start = time.time()
faiss.write_index(index, args.faiss_index)
print('Writing index took {} s'.format(time.time()-start))
