import tempfile
import unittest
from pathlib import Path

import anndata as ad
import h5py
import numpy as np

from morph2mol.prediction_io import prediction_metadata, phase1_features


class PredictionCompatibilityTests(unittest.TestCase):
    def test_new_and_existing_prediction_files(self):
        with tempfile.TemporaryDirectory() as folder:
            for meta, embedding in [('morph2mol', 'X_morph2mol_phase1'),
                                    ('voice25m', 'X_voice_phase1')]:
                path = Path(folder) / (meta + '.h5ad')
                expected = np.arange(12).reshape(3, 4).astype(np.float32)
                ad.AnnData(X=np.zeros((3, 2)), obsm={embedding: expected},
                           uns={meta: {'stage1_sha256': 'checkpoint'}}).write_h5ad(path)
                with h5py.File(path) as handle:
                    self.assertEqual(prediction_metadata(handle)['stage1_sha256'], 'checkpoint')
                    np.testing.assert_array_equal(phase1_features(handle)[:], expected)


if __name__ == '__main__':
    unittest.main()
