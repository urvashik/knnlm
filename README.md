# Nearest Neighbor Language Models

This repository is a fork of the [Fairseq](https://github.com/pytorch/fairseq) repository and the exact commit that this code is based on can be found [here](https://github.com/pytorch/fairseq/tree/6a5181509aa1fa7d260985157e77211753da544b). Please use the exact commit page to determine software requirements for using this code. This README will be updated once the code has been merged into Fairseq.

This code pertains to the ICLR 2020 paper: [Generalization through Memorization: Nearest Neighbor Language Models](https://arxiv.org/pdf/1911.00172.pdf). If you use this code or results from our paper, please cite:

```
@article{khandelwal20generalization,
  title={{Generalization through Memorization: Nearest Neighbor Language Models}},
  author={Khandelwal, Urvashi and Levy, Omer and Jurafsky, Dan and Zettlemoyer, Luke and Lewis, Mike},
  journal={International Conference on Learning Representations (ICLR)},
  year={2020}
}
```

## Wikitext-103 Experiments

Before starting, make sure you install fairseq (after pulling the code, from the project directory) and [faiss](https://github.com/facebookresearch/faiss/wiki):
```bash
pip install --editable .

pip install faiss
```

### A note about Hardware

Experiments for this paper were conducted on machines that contain 500GB of RAM, NVIDIA V100 32GB GPUs and flash storage (SSDs). Saving the Wikitext-103 datastore requires 400GB of disk space. The speed of saving the datastore, building the FAISS index and evaluating the nearest neighbors language model heavily depends on the amount of RAM available for each job. Some of these steps can be sped up by parallelizing, which we leave for users to do in order to best cater to their setup.

If you are working with a remote cluster, please note that we use [memmaps](https://numpy.org/doc/1.18/reference/generated/numpy.memmap.html) for saving the datastore. This allows us to keep the data on disk while accessing it by loading small chunks into memory, depending on the available RAM. This means there are a large number of disk seeks. In order to prevent slowing down your entire cluster, we suggest always reading/writing this data to/from local disks (as opposed to NFS directories), and flash storage is best for faster access.

### Preparing the data

We share Fairseq's instructions on how to prepare the data here.

```bash
cd examples/language_model/
bash prepare-wikitext-103.sh
cd ../..


TEXT=examples/language_model/wikitext-103
python preprocess.py \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20
```

### Training the Language Model

We share Fairseq's instructions on how to train the language model here. Alternatively, you can download the checkpoint used for our experiments [here](https://nlp.stanford.edu/projects/knnlm/wt103_checkpoint_best.pt). 

```bash
python train.py --task language_modeling \
    data-bin/wikitext-103 \
    --save-dir checkpoints/ \
    --arch transformer_lm_wiki103 \
    --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 --fp16 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d
```

This model was trained on 8 gpus.

### Evaluating the Language Model

To evaluate the model on the validation set:

```bash
python eval_lm.py data-bin/wikitext-103 \
    --path checkpoints/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid
```

### Saving the keys and values for the datastore

In order to save keys and values for the datastore, we must run model evaluation over the entire training set. 

**Caution**: Running this step requires a large amount of disk space (400GB!). Please read the note about Hardware above, before running this! 

```bash
python eval_lm.py data-bin/wikitext-103 \
    --path checkpoints/checkpoint_best.pt \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset train \
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap checkpoints/dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 103225485 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16
```

The total number of tokens in the Wikitext-103 training set is `103227021`. The dstore size `103225485` is `1536` tokens less than the total due to the context-window. We want each key to be constructed using a minimum amount of prior context.

If you would prefer to save the keys and values in float16, please use the `--dstore-fp16` flag and remember to use it during the index building and evaluation steps as well.

### Building the FAISS index

The FAISS index requires a training stage where it learns a set of clusters for the keys. Once this is completed, the keys must all be added to the index. The speed of adding keys to the index depends on the hardware, particularly the amount of RAM available. Please check the paper for more details on our use of FAISS.

Note that the following command runs on CPU.

```bash
python build_dstore.py 
    --dstore_mmap checkpoints/dstore \
    --dstore_size 103225485 \
    --faiss_index checkpoints/knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0
```

### Evaluating the Nearest Neighbor Language Model

To evaluate the model on the validation set:

```bash
python eval_lm.py data-bin/wikitext-103 \
    --path checkpoints/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024 \
    --gen-subset valid --dstore-filename checkpoints/dstore \
    --indexfile checkpoints/knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 103225485 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16
```

If your hardware constraints make this too slow, you can run it without using full precision keys by adding two flags: `--no-load-keys` and `--knn-sim-func "do_not_recomp_l2"`. This uses the quantized versions of keys stored within the FAISS index. You can make things faster by reducing the value of the `probe` (the number of clusters FAISS checks for neighbors) at the cost of performance. You can also try reducing the number of neighbors `k`.
