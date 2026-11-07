from __future__ import annotations

import unicodedata

from .common import DocscanError


def normalize_text(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def character_errors(reference: str, prediction: str) -> dict:
    """Count one minimum-cost alignment; ties prefer diagonal, deletion, then insertion."""
    n = len(reference)
    if not n:
        raise DocscanError("empty_ground_truth", "Normalized reference text is empty")
    # Equal edges can be stripped without changing the minimum edit distance.
    prefix = 0
    for left, right in zip(reference, prediction):
        if left != right:
            break
        prefix += 1
    reference, prediction = reference[prefix:], prediction[prefix:]
    suffix = 0
    for left, right in zip(reversed(reference), reversed(prediction)):
        if left != right:
            break
        suffix += 1
    if suffix:
        reference, prediction = reference[:-suffix], prediction[:-suffix]

    # Each cell stores (cost, substitutions, deletions, insertions).
    previous = [(j, 0, 0, j) for j in range(len(prediction) + 1)]
    for i, left in enumerate(reference, start=1):
        current = [(i, 0, i, 0)]
        for j, right in enumerate(prediction, start=1):
            change = int(left != right)
            cost, sub, delete, insert = previous[j - 1]
            best = (cost + change, sub + change, delete, insert)
            cost, sub, delete, insert = previous[j]
            if cost + 1 < best[0]:
                best = (cost + 1, sub, delete + 1, insert)
            cost, sub, delete, insert = current[-1]
            if cost + 1 < best[0]:
                best = (cost + 1, sub, delete, insert + 1)
            current.append(best)
        previous = current
    edits, sub, delete, insert = previous[-1]
    return {"S": sub, "D": delete, "I": insert, "N": n, "cer": edits / n}
