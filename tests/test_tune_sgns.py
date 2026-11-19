import csv

import tune_sgns

FIELDS = tune_sgns.KNOBS + ["w+c", "seed"] + tune_sgns.METRICS
A = {"negative": "5", "ns_exponent": "0.75", "sample": "0.001", "vector_size": "100", "epochs": "5"}
B = {**A, "epochs": "15"}


def row(config, wc):
    return {**config, "w+c": str(wc), "seed": "1", "mrr": "0.1", "hit@1": "0", "hit@3": "0",
            "hit@10": "0", "mrr_covered": "0.1", "coverage": "1", "n": "3"}


def write(path, rows, tail=""):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
        f.write(tail)


def test_resume_keeps_only_configs_with_both_rows_and_drops_a_cut_row(tmp_path):
    # A run killed mid-write leaves B with only its W row and a half-written line at the end.
    path = tmp_path / "sgns_val.csv"
    write(path, [row(A, False), row(A, True), row(B, False)], tail="15,0.75,0.0")
    done = tune_sgns.keep_complete_configs(path)
    assert done == {tune_sgns.config_key(A)}
    rows = tune_sgns.read_csv(path)
    assert [(r["epochs"], r["w+c"]) for r in rows] == [("5", "False"), ("5", "True")]
    assert path.read_text().endswith("\n")  # new rows can be appended cleanly


def test_complete_file_is_left_as_it_is(tmp_path):
    path = tmp_path / "sgns_val.csv"
    write(path, [row(A, False), row(A, True)])
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    assert tune_sgns.keep_complete_configs(path) == {tune_sgns.config_key(A)}
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before  # not even rewritten


def test_last_line_cut_inside_its_last_value_is_dropped(tmp_path):
    # Every column is there but "n" was cut from 3 digits to 1 and the newline never came.
    path = tmp_path / "sgns_val.csv"
    write(path, [row(A, False), row(A, True), row(B, False)])
    cut = ",".join(row(B, True)[k] for k in FIELDS)[:-1] + "1"
    with open(path, "a", newline="") as f:
        f.write(cut)
    assert tune_sgns.keep_complete_configs(path) == {tune_sgns.config_key(A)}


def test_empty_results_file_starts_over(tmp_path):
    path = tmp_path / "sgns_val.csv"
    path.write_text("")
    assert tune_sgns.keep_complete_configs(path) == set()
    assert not path.exists()  # the grid then writes a fresh header
