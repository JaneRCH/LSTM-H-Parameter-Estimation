"""Deterministic seed derivation from a single master seed.

All randomness in this study descends from one number, the student number
56233043, through :class:`numpy.random.SeedSequence`.  Passing the master seed
directly to every generator would make the draws dependent: the training
corpus, the validation set and the Monte Carlo nulls would share a stream, and
in the worst case a test set would duplicate a training set.  Spawning named
children avoids that while keeping the whole study reproducible from the single
number recorded in the report.

Example
-------
>>> rng = generator("corpus_train")
>>> int(rng.integers(0, 1_000_000))
418155
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["MASTER_SEED", "entropy_for", "generator", "seed_table"]

MASTER_SEED: int = 56233043
"""Student number of the author, used as the single root of all randomness."""

#: Streams used anywhere in the study.  Adding a name here is the only
#: supported way to obtain randomness: it keeps the registry auditable and
#: prevents two components silently sharing a stream.
_STREAMS: tuple[str, ...] = (
    "corpus_train",
    "corpus_test",
    "corpus_val",
    "net_init",
    "net_batch",
    "identification",
    "prior_predictive",
    "bootstrap",
    "nulls",
    "surrogates",
    "episodes",
    "out_of_family",
    "power_lambda",
    "tests",
)


def _child_index(name: str) -> int:
    """Map a stream name to a stable, collision-resistant child index.

    The index is the first eight bytes of the SHA-256 digest of the name.  It
    does not depend on the position of the name in :data:`_STREAMS`, so adding
    or reordering streams never changes an existing stream's draws.
    """
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def entropy_for(name: str) -> np.random.SeedSequence:
    """Return the :class:`SeedSequence` for a named stream.

    Raises
    ------
    KeyError
        If ``name`` is not a registered stream.
    """
    if name not in _STREAMS:
        raise KeyError(
            f"unknown stream {name!r}; register it in rvmf.seeds._STREAMS. "
            f"Known: {sorted(_STREAMS)}"
        )
    return np.random.SeedSequence(entropy=MASTER_SEED, spawn_key=(_child_index(name),))


def generator(name: str, replicate: int | None = None) -> np.random.Generator:
    """Return a fresh generator for a named stream.

    Parameters
    ----------
    name
        A registered stream name.
    replicate
        Optional replicate index.  Two replicates of the same stream are
        independent, which is what Monte Carlo loops need.
    """
    seq = entropy_for(name)
    if replicate is not None:
        seq = seq.spawn(replicate + 1)[replicate]
    return np.random.default_rng(seq)


def seed_table() -> dict[str, int]:
    """Report the first 32-bit word of each stream, for the audit table."""
    return {
        name: int(entropy_for(name).generate_state(1, dtype=np.uint32)[0])
        for name in _STREAMS
    }
