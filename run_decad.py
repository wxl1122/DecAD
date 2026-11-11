"""Command-line entry point for DecAD training and evaluation."""
import os

# Use PyTorch SDPA; no compiled xFormers extension is required.
os.environ.setdefault('XFORMERS_DISABLED', '1')
# Avoid unnecessary transparent-huge-page compaction for large NumPy caches.
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

from decad.runner import main

if __name__ == '__main__':
    main()
