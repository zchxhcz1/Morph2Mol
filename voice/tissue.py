"""Resolve corpus objects and read user-supplied organ annotations."""
from pathlib import Path
import csv
import re


def tissue_key(name):
    return (name or "").strip().casefold()


def object_directory(dataset, sample):
    root = Path(dataset)
    matches = sorted(p for p in root.glob(f"*/*/*/{sample}")
                     if p.is_dir() and "rawdata" not in p.parts[-3].casefold())
    if not matches:
        raise FileNotFoundError(f"{sample}: no object under {root}")
    if len({p.resolve() for p in matches}) != 1 or len({tissue_key(p.parent.name) for p in matches}) != 1:
        raise ValueError(f"{sample}: ambiguous corpus objects: {matches}")
    return next((p for p in matches if p.relative_to(root).parts[0] == "training_set"), matches[0])


def read_tissue_map(path):
    """Read sample,tissue CSV; an empty tissue includes a reference only in global fitting."""
    tissues = {}
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'sample', 'tissue'} <= set(reader.fieldnames or []):
            raise ValueError('Tissue map requires sample and tissue columns')
        for row in reader:
            sample = (row['sample'] or '').strip()
            tissue = tissue_key(row['tissue'])
            if not sample or sample in tissues:
                raise ValueError(f'Tissue map has an empty or duplicate sample: {sample!r}')
            if tissue and not re.fullmatch(r'[a-z][a-z0-9_]*', tissue):
                raise ValueError(f'Use a lowercase organ name with underscores: {tissue!r}')
            tissues[sample] = tissue
    return tissues


EXCLUSION_FIELDS = ('donor_or_block_id',)


def reference_exclusions(dataset, sample, bank_objects, fields=EXCLUSION_FIELDS):
    """Leave shared donor/block groups out when fitting a new split's beta."""
    excluded = {sample}
    groups_path = Path(dataset)/'metadata/shared_origin_groups.csv'
    if groups_path.is_file():
        with groups_path.open() as f:
            groups = [set(row['samples'].split(';')) for row in csv.DictReader(f)
                      if row.get('field') in fields]
        while True:
            expanded = excluded | set().union(*(g for g in groups if g & excluded))
            if expanded == excluded:
                break
            excluded = expanded
    return sorted(excluded & set(bank_objects))
