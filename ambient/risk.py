"""
Risk tier constants.

A module of its own so that `gate` and `tools` can both import them without
importing each other. gate needs the tiers to decide; tools needs them to
declare. Neither should own the vocabulary.

    SAFE     read-only. Runs silently.
    CAUTION  reversible state change. Runs, and is written to audit.log.
    DANGER   destructive or irreversible. Stops and asks a human first.
"""

SAFE = "SAFE"
CAUTION = "CAUTION"
DANGER = "DANGER"

ALL = (SAFE, CAUTION, DANGER)
