# DART evaluation of a LoRA-tuned GPT-2 Medium (nanoGPT)

This repo is a stripped-down fork of the GPT-2 part of [microsoft/LoRA](https://github.com/microsoft/LoRA)
(**LoRA: Low-Rank Adaptation of Large Language Models**, Hu et al., https://arxiv.org/abs/2106.09685).
It keeps only the DART evaluation pipeline (beam search, decoding, metrics) and runs it on a GPT-2 Medium
checkpoint trained with [nano_lora](https://github.com/panalexeu/nano_lora), with the LoRA adapters already merged into the weights.

* [src/](src): data conversion, encoding, beam search and decoding
* [eval/](eval): the patched `eval.py` and a script that downloads the metric code (GenerationEval)
* [data/dart/](data/dart): raw DART v1.1.1 json files, the converted test set and its encoded version
* [vocab/](vocab): GPT-2 BPE vocabulary (`encoder.json`, `vocab.bpe`)
* [eval_out/](eval_out): decoded hypothesis and reference files of the reported run

## Requirements

* Python 3.12 and a CUDA GPU with bf16 support (the reported run used an RTX 3050 Laptop, 4 GB)
* `java` (METEOR) and `perl` (BLEU) on the `PATH`

## Evaluating on DART from scratch

All commands run from the repo root.

1. Install the dependencies:
```
python -m venv .venv
. .venv/bin/activate
pip install -r requirement.txt
```

2. Put the checkpoint at `./dart_gpt.pt`. It is a `torch.save` dict with:
   * `model`: the nanoGPT `state_dict` with LoRA merged in (a `_orig_mod.` prefix from `torch.compile` is stripped on load)
   * `model_args`: the `GPTConfig` kwargs (`n_layer=24, n_head=16, n_embd=1024, block_size=1024, vocab_size=50257, bias=True`)

3. Convert the DART test set to the training prompt format (`### Triples:\n...\n\n### Text:\n`), then encode it with the GPT-2 BPE.
   Both outputs are already committed, so you can skip this step.
```
python src/format_converting_dart.py data/dart/dart-v1.1.1-full-test.json data/dart/test.jsonl
python src/gpt2_encode.py --vocab vocab --input data/dart/test.jsonl --output data/dart/test_enc.jsonl --add_eos
```

4. Run beam search (beam 10, length penalty 0.8, no-repeat 4-gram, as in the paper). Each unique triple set is decoded once.
   `--batch_size 6` fits in 4 GB; the KV cache is allocated once for `batch_size * beam` rows, so lower it if you hit OOM at startup.
```
torchrun --nproc_per_node=1 src/gpt2_beam.py \
    --data ./data/dart/test_enc.jsonl \
    --batch_size 6 \
    --seq_len 256 \
    --eval_len 64 \
    --init_checkpoint ./dart_gpt.pt \
    --beam 10 \
    --length_penalty 0.8 \
    --no_repeat_ngram_size 4 \
    --repetition_penalty 1.0 \
    --work_dir ./data/dart \
    --output_file dart_predict.jsonl \
    --platform local
```
   The EOS token defaults to `50256` (`<|endoftext|>`), which is how completions ended in training.

5. Decode the predictions into the hypothesis and reference files (one line per unique triple set, up to 6 references each):
```
mkdir -p eval_out
python src/gpt2_decode.py \
    --vocab ./vocab \
    --sample_file ./data/dart/dart_predict.jsonl \
    --input_file ./data/dart/test.jsonl \
    --ref_type dart \
    --ref_num 6 \
    --output_ref_file ./eval_out/references_dart \
    --output_pred_file ./eval_out/hypothesis_dart \
    --tokenize --lower
```

6. Download the metric code (GenerationEval, e2e-metrics, METEOR, BLEURT). This also moves the patched `eval/eval.py` into `eval/GenerationEval/`.
   The versions GenerationEval pins don't work on Python 3.12, so upgrade them afterwards and fetch the NLTK tokenizer data:
```
bash eval/download_evalscript.sh
pip install "nltk==3.10.4" "tabulate==0.9.0"
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

7. Compute BLEU, METEOR and TER:
```
cd eval/GenerationEval
python eval.py -R ../../eval_out/references_dart/reference -H ../../eval_out/hypothesis_dart -nr 6 -m bleu,meteor,ter
cd ../..
```

## Results

DART test set (4,159 unique triple sets), beam 10. The paper's numbers are from **Appendix F.1, Table 13** of the LoRA paper.

| GPT-2 Medium + LoRA | BLEU ↑ | MET ↑ | TER ↓ |
|---|---|---|---|
| LoRA paper (Table 13) | 47.1±.2 | 0.39 | 0.46 |
| This checkpoint (`dart_gpt.pt`) | 47.41 | 0.39 | 0.47 |

## Changes compared to microsoft/LoRA

* Removed training/fine-tuning code, the `loralib` model, and the E2E/WebNLG data and converters; only DART evaluation is kept.
* Replaced the model definition in `src/model.py` with nanoGPT's (the architecture used for training).
* Added an optional KV cache to `CausalSelfAttention`: prefill writes the prompt, and each decode step writes one token per row at its own position, with a mask over the right-padding slots. Without a cache the model behaves as before.
* `src/gpt2_beam.py`:
  * loads the nanoGPT checkpoint (`model` + `model_args`) on CPU, then moves the model to the GPU in bf16 and runs under bf16 autocast (logits are scored in fp32)
  * beam search on the KV cache: the cache is allocated once per run, each prompt is prefilled once and copied to its beams in place, and the cache is reordered layer by layer after each beam selection
  * decodes each unique context once, since `gpt2_decode.py` keeps a single prediction per context (4,159 instead of 12,552)
  * exact early stop: a batch stops once no running beam can beat any example's best finished output
  * prints the time per batch and the number of steps
* `src/format_converting_dart.py`: uses the training prompt template `### Triples:\n{triples}\n\n### Text:\n`.
* `src/gpt2_encode.py`: removed the `--add_bos` option (no separator token after the context, matching training) and unused `numpy` imports; `create_datasets.sh` no longer passes `--add_bos`.
* `src/gpt2_decode.py`: removed an unused `numpy` import.
* `eval/eval.py` (GenerationEval):
  * imports BLEURT only when `-m bleurt` is requested, so it doesn't need TensorFlow otherwise
  * drops the trailing newline when reading the reference and hypothesis files; before, it added an empty extra example and broke METEOR
* `requirement.txt`: pinned torch 2.14 / CUDA 13 wheels and added `regex` (needed by the GPT-2 BPE encoder).
* `.gitignore`: ignores checkpoints (`*.pt`), the venv and the cloned metric repos.

## Acknowledgements

The evaluation pipeline was ported with help from [Claude Code](https://claude.com/claude-code) (Claude Opus 5.5, Anthropic).
It worked through the tedious parts: getting beam search to run with the nanoGPT model, the KV cache,
fitting the eval on a 4 GB GPU, and the dependency and parsing fixes in GenerationEval.

## Citation
```
@misc{hu2021lora,
    title={LoRA: Low-Rank Adaptation of Large Language Models},
    author={Hu, Edward and Shen, Yelong and Wallis, Phil and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Lu and Chen, Weizhu},
    year={2021},
    eprint={2106.09685},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
