"""Read current predictions and previously exported analysis results."""


def prediction_metadata(h5):
    from anndata.io import read_elem
    key = 'morph2mol' if 'morph2mol' in h5['uns'] else 'voice25m'
    return read_elem(h5['uns'][key])


def phase1_features(h5):
    key = 'X_morph2mol_phase1' if 'X_morph2mol_phase1' in h5['obsm'] else 'X_voice_phase1'
    return h5['obsm'][key]
