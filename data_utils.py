"""Helpers for NVIDIA OpenCodeInstruct prompt/completion preprocessing."""

def valid_record(row, min_score=None):
    if not isinstance(row, dict):
        return False
    if not isinstance(row.get("input"), str) or not row["input"].strip():
        return False
    if not isinstance(row.get("output"), str) or not row["output"].strip():
        return False
    if min_score is not None:
        try:
            return float(row["average_test_score"]) >= min_score
        except (KeyError, TypeError, ValueError):
            return False
    return True

def record_to_conversation(row):
    return {
        "prompt": [{"role": "user", "content": row["input"].strip()}],
        "completion": [{"role": "assistant", "content": row["output"].strip()}],
    }

def select_disjoint_rows(stream, train_count, eval_count, min_score=None):
    seen = set()
    selected = []
    for row in stream:
        if not valid_record(row, min_score):
            continue
        identifier = row.get("id")
        if identifier is not None:
            if identifier in seen:
                continue
            seen.add(identifier)
        selected.append(row)
        if len(selected) >= train_count + eval_count:
            break
    if len(selected) < train_count + eval_count:
        raise ValueError("Not enough qualifying rows")
    return selected[eval_count:], selected[:eval_count]
