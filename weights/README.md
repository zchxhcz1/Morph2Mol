# Model files

Download the model from [Hugging Face](https://huggingface.co/zchx/Morph2Mol). Obtain `uni2h/pytorch_model.bin` separately through [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h), using an account with approved access.

The complete directory used by `--release` contains:

```text
morph2mol/
  release.json
  stage1.pt
  stage2.pt
  gene_space.npz
  xenium_genes.tsv
  beta.json
  beta.meta.json
  beta/
    manifest.json
    lung/narrow.json
    lung/wide.json
    ...
  uni2h/pytorch_model.bin
  bank/
    model.json
    bank_complete.json
    summaries/object_00/...
    ...
```

`release.json` supplies relative paths to the model files. Keep the checkpoints, gene information, fusion weights, and bank together. `genes.tsv` and `xenium_genes.tsv` in this source repository list model gene symbols, one per line without a header.

The root `beta.json` contains all organ/panel groups and global fallback weights and is used automatically by prediction. `beta/manifest.json` lists each available group, reference count, fitted-gene count, and checksum; `beta/ORGAN/WIDTH.json` contains the corresponding gene-wise weights. Select a group with `--tissue lung/wide` or supply an organ such as `--tissue lung` for automatic width selection.

To package a newly trained model:

```bash
python train/export_release.py \
  --uni2h /models/uni2h \
  --stage1 /runs/morph2mol/checkpoints/alignment/stage1_final.pt \
  --stage2 /runs/morph2mol/checkpoints/expression/stage2_last.pt \
  --space /runs/morph2mol/direct/gene_space_all.npz \
  --xenium-space /runs/morph2mol/direct/gene_space_xen.npz \
  --bank /runs/bank --beta /runs/beta.json \
  --out /models/morph2mol
```

The destination must be empty. Omit both `--bank` and `--beta` when exporting the intermediate checkpoints used by reference preparation during model training. Follow the UNI2-h access and distribution terms when sharing model files.
