#!/usr/bin/env python
"""Build gene embeddings for the union of genes measured in the training corpus."""
from pathlib import Path
import argparse
import csv
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def training_gene_union(dataset):
    from morph2mol.training_contract import read_ids
    from morph2mol.tissue import object_directory
    root = Path(dataset)
    samples = read_ids(root / 'splits/training_set.txt')
    if not samples:
        raise ValueError('The training split is empty')
    genes = set()
    for sample in samples:
        folder = object_directory(root, sample)
        if folder.relative_to(root).parts[0] != 'training_set':
            raise ValueError(f'{sample}: expected a training-set object')
        with (folder / 'genes.tsv').open(newline='') as stream:
            reader = csv.reader(stream, delimiter='\t')
            if not next(reader, None):
                raise ValueError(f'{sample}: empty gene table')
            symbols = [row[0].strip() for row in reader if row]
        if not symbols or not all(symbols) or len(symbols) != len(set(symbols)):
            raise ValueError(f'{sample}: genes must be unique and nonempty')
        genes.update(symbols)
    return sorted(genes)


def gene_embeddings(genes, state, vocab):
    import numpy as np
    import torch
    w = state['encoder.embedding.weight'].float()
    norm = torch.nn.LayerNorm(w.shape[1])
    norm.weight.data = state['encoder.enc_norm.weight'].float()
    norm.bias.data = state['encoder.enc_norm.bias'].float()
    upper = {k.upper(): v for k, v in vocab.items()}
    ids = [vocab.get(g, upper.get(g.upper(), -1)) for g in genes]
    covered = np.array([i >= 0 for i in ids])
    emb = np.zeros((len(genes), w.shape[1]), np.float32)
    if covered.any():
        with torch.no_grad():
            emb[covered] = norm(w[torch.tensor([i for i in ids if i >= 0])]).numpy()
    return emb, covered


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--dataset',help='Prepared corpus; uses the sorted union of all training-set genes')
    source.add_argument('--genes',help='Explicit ordered symbols for reproducing an existing gene head')
    p.add_argument('--scgpt-dir',required=True,help='best_model.pt and vocab.json')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    import numpy as np
    import torch
    genes=(training_gene_union(a.dataset) if a.dataset else
           [g.strip() for g in Path(a.genes).read_text().splitlines() if g.strip()])
    if not genes or len(set(genes))!=len(genes):raise ValueError('Genes must be unique and nonempty')
    root=Path(a.scgpt_dir)
    state=torch.load(root/'best_model.pt',map_location='cpu',weights_only=False)
    vocab=json.loads((root/'vocab.json').read_text())
    emb,covered=gene_embeddings(genes,state,vocab)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,genes=np.array(genes),emb=emb,covered=covered)


if __name__=='__main__':main()
