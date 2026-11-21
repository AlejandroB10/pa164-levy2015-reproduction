import pytest

from sample_parser_lines import score

HEADER = "line\tparsed\tcorrect\texpected\tnotes\n"


def write(tmp_path, rows):
    path = tmp_path / "sample.tsv"
    path.write_text(HEADER + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_score_counts_only_labelled_rows(tmp_path, capsys):
    path = write(tmp_path, [
        ("1 cup flour", "flour", "y", "", ""),
        ("1 cup AP flour", "flour", "n", "all_purpose_flour", ""),
        ("2 eggs", "egg", "", "", ""),
    ])
    score(path)
    out = capsys.readouterr().out
    assert "labelled 2; accuracy 50.0%" in out
    assert "(2 matched)" in out


def test_score_skips_an_unlabelled_row_with_its_trailing_tabs_stripped(tmp_path, capsys):
    path = tmp_path / "sample.tsv"
    path.write_text(HEADER + "1 cup flour\tflour\ty\t\t\n2 eggs\tegg\n", encoding="utf-8")
    score(path)
    assert "labelled 1; accuracy 100.0%" in capsys.readouterr().out


def test_score_rejects_a_mistyped_label(tmp_path):
    path = write(tmp_path, [
        ("1 cup flour", "flour", "y", "", ""),
        ("2 eggs", "egg", "Y", "", ""),
    ])
    with pytest.raises(SystemExit, match="found \\['Y'\\]"):
        score(path)
