# Training

Training consists of preparing RNA embeddings, aligning image and RNA features, fitting the expression model, building a reference bank, and fitting gene-wise fusion weights.

```bash
pip install -r requirements-train.txt
```

Use CUDA for model training. Commands below support one GPU with `python` or multiple GPUs with `torchrun`. Both stages use the fixed three-epoch Morph2Mol recipe. Runtime options control paths, workers, checkpoint frequency, encoder chunk size and continuation.

All paths below are examples; use your own data, model and output directories.

## Prepare the corpus

Prepare each slide as described in [formats.md](formats.md):

```text
corpus/
  MANIFEST.csv
  training_set/{xenium,cosmx}/TISSUE/SAMPLE/...
  cross_slide_test_set/{xenium,cosmx}/TISSUE/SAMPLE/...
  splits/
    training_set.txt
    inslide_test_set.txt
    cross_slide_test_set.txt
    stage2_global_training.txt
    retrieval_references.txt
    gate_fit_references.txt
  metadata/shared_origin_groups.csv
```

Each split file contains one globally unique sample name per line. Image–RNA alignment uses `training_set.txt`. The expression-training set equals the training set minus the in-slide test samples. Cross-slide test samples are disjoint from the training set. `retrieval_references.txt` and `gate_fit_references.txt` contain the expression-training samples. Empty test splits are allowed. `MANIFEST.csv` records the corpus used for training.

The optional `shared_origin_groups.csv` identifies related donor or tissue-block samples. It contains `field,samples`, with `field=donor_or_block_id` and semicolon-separated sample names. Related samples are excluded when fitting fusion weights for a reference slide.

## Prepare RNA embeddings

Obtain the scGPT whole-human files `best_model.pt`, `vocab.json`, and `args.json` from [scGPT](https://github.com/bowang-lab/scGPT). Generate cell embeddings in expression-matrix row order:

```bash
python train/embed_scgpt.py --dataset /data/corpus \
  --scgpt-dir /models/scgpt_human --out /runs/scgpt_cells
```

The extractor uses nonzero genes, a classification token, 51 expression bins, at most 1,200 tokens, and normalized 512-dimensional cell features. `--only sample_a,sample_b` and sharding options support processing subsets.

Create the gene embeddings and model gene space for the released 18,580-gene prediction head:

```bash
python train/build_gene_table.py \
  --scgpt-dir /models/scgpt_human --out /runs/gene_table.npz
python train/prepare.py --dataset /data/corpus --work-dir /runs/morph2mol \
  --gene-table /runs/gene_table.npz
```

By default, the first command reads the repository's `weights/genes.tsv` in file order. This is equivalent to passing `--genes weights/genes.tsv` from the repository root. Gene embeddings use scGPT's embedding layer followed by encoder layer normalization. Symbols are matched exactly, then without case sensitivity. Genes outside the scGPT vocabulary retain their output positions and start with zero embeddings that the expression head can learn during training.

For a custom head, supply an ordered list with `--genes FILE` or derive one from a prepared training split with `--dataset DIR`. A custom vocabulary requires a matching prediction head, reference bank and fitted fusion weights.

## Align image and RNA features

```bash
torchrun --standalone --nproc_per_node=3 train/train_phase1.py \
  --dataset /data/corpus --work-dir /runs/morph2mol \
  --uni2h /models/uni2h --cell-embeddings /runs/scgpt_cells
```

| Setting | Value |
| --- | --- |
| Training duration | 3 epochs, all usable cells, retained tail batches |
| Objective | Symmetric contrastive loss, temperature 0.07 |
| Optimizer | AdamW, weight decay 0.01 |
| Learning rates | Image adapters 1.5e-4; projection towers 1e-3 |
| Image adapters | Last 12 encoder blocks, rank 16, alpha 32, dropout 0.05 |
| Augmentation | Random rotations and reflections |
| Batch | 2,048 cells per rank, encoder chunks of 256 |

Output: `/runs/morph2mol/checkpoints/alignment/stage1_final.pt`. Adjust `--chunk` for your hardware; it must divide the fixed effective batch size of 2,048 cells per rank. `--resume` restores weights, optimizer state, and data position; use the same world size and corpus when resuming.

## Train expression prediction

```bash
torchrun --standalone --nproc_per_node=3 train/train_phase2.py \
  --dataset /data/corpus --work-dir /runs/morph2mol --uni2h /models/uni2h \
  --stage1 /runs/morph2mol/checkpoints/alignment/stage1_final.pt
```

| Setting | Value |
| --- | --- |
| Training duration | 3 epochs with complete target-cell coverage |
| Spatial decoder | 6 layers, 256 features, maximum distance 137 µm |
| Neighborhood | Up to 128 supervised cells and 64 additional context cells |
| Learning rates | Image adapters 3e-5; decoder and expression head 1e-4 |
| Expression head | Learnable gene embeddings initialized from scGPT |
| Objective | Negative-binomial loss plus log1p mean squared error |
| Encoder updates | First 3 adapted blocks frozen; remaining adapters trainable |
| Augmentation | Disabled during expression training |

Output: `/runs/morph2mol/checkpoints/expression/stage2_last.pt`. Each epoch rotates and shifts the spatial partition, rebuilds target blocks, and shuffles their order. Every usable target cell contributes loss exactly once per epoch. Context cells participate in spatial attention but do not contribute expression loss. Each slide supervises only its measured genes. Gradient checkpointing is enabled; `--resume` restores the complete training state. Prediction reuses the same target partition and neighbor-selection functions with a fixed seed and partition epoch.

## Build the reference bank

Package the intermediate model, encode the image-alignment training slides, and construct the bank:

```bash
python train/export_release.py --uni2h /models/uni2h \
  --stage1 /runs/morph2mol/checkpoints/alignment/stage1_final.pt \
  --stage2 /runs/morph2mol/checkpoints/expression/stage2_last.pt \
  --space /runs/morph2mol/direct/gene_space_all.npz \
  --xenium-space /runs/morph2mol/direct/gene_space_xen.npz \
  --out /models/morph2mol_direct
python train/prepare_bank.py --release /models/morph2mol_direct \
  --dataset /data/corpus --references /data/corpus/splits/training_set.txt \
  --out /runs/reference_inputs
python predict/build_bank.py --manifest /runs/reference_inputs/manifest.json \
  --output /runs/bank --device cuda --backend torch
```

Bank construction uses PCA with 64 components, fitted on up to 100,000 cells per slide, followed by 10 k-means iterations. The initial number of clusters is `max(1, floor(N/100))`; empty clusters are removed. Each reference requires at least 64 cells. The backend can be `torch` or GPU-enabled `faiss`.

Retrieval uses cosine similarity in the 1,536-dimensional image feature space, 50 neighboring clusters per measured-gene coverage group, and exponential weights with temperature 0.03. Each reference contributes at most its 50 closest clusters. Ties at the selection threshold share the remaining weight within this candidate set. Cluster counts do not weight expression retrieval.

## Fit fusion weights

```bash
python train/fit_beta.py --release /models/morph2mol_direct --bank /runs/bank \
  --dataset /data/corpus --references /data/corpus/splits/gate_fit_references.txt \
  --tissue-map /data/corpus/tissues.csv \
  --reference-inputs /runs/reference_inputs --out /runs/beta.json
python train/export_release.py --uni2h /models/uni2h \
  --stage1 /runs/morph2mol/checkpoints/alignment/stage1_final.pt \
  --stage2 /runs/morph2mol/checkpoints/expression/stage2_last.pt \
  --space /runs/morph2mol/direct/gene_space_all.npz \
  --xenium-space /runs/morph2mol/direct/gene_space_xen.npz \
  --bank /runs/bank --beta /runs/beta.json --out /models/morph2mol
```

Fusion weights are fitted on reference slides, excluding each reference and related donor/block samples from its retrieval bank. Calibration uses up to 20,000 deterministically sampled cells per reference by default (`--max-cells`, `--seed`). Each gene's weight is selected from 41 values between 0 and 1 by prediction correlation with measured log1p expression, then averaged over eligible references with equal reference weight.

Supply `--tissue-map` as a CSV with `sample,tissue` columns and one row for every fitting reference. Use canonical organ names such as `lung`, `colon`, `liver` and `ovary`; annotate each sample by its organ even when a source dataset contains several organs. Organ labels are supplied by the user rather than inferred from dataset or platform directory names. An empty `tissue` field includes that reference in the global fit only.

```csv
sample,tissue
lung_reference_a,lung
lung_reference_b,lung
colon_reference_a,colon
```

Weights are grouped as `ORGAN/narrow` or `ORGAN/wide`, using a 1,000-measured-model-gene threshold. Every nonempty group is exported, including single-reference groups. Retrieval for fitting uses the full reference bank excluding the query reference and related donor/block samples, so a single reference of an organ still has other references available for retrieval. Per-gene weights are averaged equally across contributing reference slides within each group.

Prediction first uses matching organ and width, then another available width of the same organ, then global weights. `--tissue lung/wide` fixes the width; `--tissue lung` selects width from the requested gene count. Organ names are compared without case sensitivity. Target-slide expression is not used during prediction.
