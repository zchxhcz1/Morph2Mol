import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from train.build_gene_table import training_gene_union


class GeneTableTests(unittest.TestCase):
    def test_training_union_keeps_uncovered_genes_and_excludes_test_only_genes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'splits').mkdir()
            (root / 'splits/training_set.txt').write_text('slide_b\nslide_a\n')
            for split, sample, genes in [
                ('training_set', 'slide_a', ['A', 'OUTSIDE_VOCAB']),
                ('training_set', 'slide_b', ['B', 'A']),
                ('cross_slide_test_set', 'slide_test', ['TEST_ONLY']),
            ]:
                d = root / split / 'xenium' / 'lung' / sample
                d.mkdir(parents=True)
                (d / 'genes.tsv').write_text('gene_symbol\n' + '\n'.join(genes) + '\n')
            self.assertEqual(training_gene_union(root), ['A', 'B', 'OUTSIDE_VOCAB'])
            scgpt = root / 'scgpt'
            scgpt.mkdir()
            (scgpt / 'vocab.json').write_text(json.dumps({'a': 0, 'B': 1}))
            torch.save({
                'encoder.embedding.weight': torch.arange(1024).reshape(2, 512).float(),
                'encoder.enc_norm.weight': torch.ones(512),
                'encoder.enc_norm.bias': torch.zeros(512),
            }, scgpt / 'best_model.pt')
            out = root / 'table.npz'
            subprocess.run([
                sys.executable, 'train/build_gene_table.py', '--dataset', str(root),
                '--scgpt-dir', str(scgpt), '--out', str(out),
            ], check=True, cwd=Path(__file__).resolve().parents[1])
            with np.load(out) as result:
                self.assertEqual(result['genes'].tolist(), ['A', 'B', 'OUTSIDE_VOCAB'])
                np.testing.assert_array_equal(result['covered'], [True, True, False])
                self.assertEqual(result['emb'].shape, (3, 512))
                np.testing.assert_array_equal(result['emb'][2], np.zeros(512))
                self.assertGreater(np.linalg.norm(result['emb'][0]), 0)

    def test_training_split_cannot_resolve_to_a_held_out_slide(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'splits').mkdir()
            (root / 'splits/training_set.txt').write_text('test_slide\n')
            sample = root / 'cross_slide_test_set/xenium/lung/test_slide'
            sample.mkdir(parents=True)
            (sample / 'genes.tsv').write_text('gene_symbol\nA\n')
            with self.assertRaisesRegex(ValueError, 'training-set'):
                training_gene_union(root)


if __name__ == '__main__':
    unittest.main()
