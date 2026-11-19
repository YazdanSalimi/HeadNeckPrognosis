"""Small helper to run a function over many inputs in parallel with a progress bar."""

import multiprocessing
import os

from tqdm import tqdm


def run_parallel(function, inputs, num_workers=-1, desc="Processing"):
    """
    Apply ``function`` to every item of ``inputs`` using a process pool.

    Parameters
    ----------
    function : callable taking a single argument (must be importable / top-level)
    inputs : iterable of arguments
    num_workers : int, -1 means "use all CPU cores"
    desc : label shown on the progress bar

    Returns
    -------
    list of results (in completion order, not input order)
    """
    inputs = list(inputs)
    if not inputs:
        print("Nothing to process: the input list is empty.")
        return []

    max_workers = min(len(inputs), os.cpu_count() or 1)
    num_workers = max_workers if num_workers == -1 else max(1, min(num_workers, max_workers))

    with multiprocessing.get_context("spawn").Pool(processes=num_workers) as pool:
        return list(
            tqdm(
                pool.imap_unordered(function, inputs),
                total=len(inputs),
                desc=f"{desc} ({num_workers} workers)",
            )
        )
