#!/usr/bin/env python
"""Train Morph2Mol-25M stage 2 with the fixed three-epoch recipe."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train._entry import arguments


def main():
    paths = arguments(2)
    from morph2mol.training.stage2 import main as run
    run(paths)


if __name__ == '__main__':
    main()
