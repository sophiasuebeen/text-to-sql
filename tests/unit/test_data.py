import json

import pytest

from text_to_sql.data import (
    IGNORE_INDEX,
    Example,
    SplitError,
    TemplateContractError,
    assign_splits,
    build_database_manifest,
    build_supervised_example,
    check_disjoint,
    load_spider_file,
    pad_batch,
    prompt_fits,
    select_dev_subset,
    split_databases,
)

from fakes import FakeTokenizer


def rows(dbs_counts, prefix):
    out, i = [], 0
    for db, n in dbs_counts:
        for _ in range(n):
            out.append({"id": f"{prefix}-{i:05d}", "db_id": db, "question": f"q{i}", "query": "SELECT 1", "source": f"f#{i}"})
            i += 1
    return out


def test_load_spider_file_stable_ids(tmp_path):
    p = tmp_path / "train_spider.json"
    p.write_text(json.dumps([{"db_id": "a", "question": "Q?", "query": "SELECT 1", "extra": 1}] * 2))
    r = load_spider_file(p, "train")
    assert [x["id"] for x in r] == ["train-00000", "train-00001"]
    assert r[1]["source"] == "train_spider.json#1"


def test_split_databases_deterministic_and_disjoint():
    dbs = [f"db{i}" for i in range(25)]
    tr, dv = split_databases(dbs + dbs, 0.1, seed=5)
    assert (tr, dv) == split_databases(list(reversed(dbs)), 0.1, seed=5)
    assert len(dv) == 3 and not set(tr) & set(dv) and len(tr) + len(dv) == 25


def test_assign_splits_keeps_databases_whole():
    train = rows([(f"d{i}", 3) for i in range(20)], "train")
    test = rows([("t1", 2), ("t2", 2)], "test")
    ex = assign_splits(train, test, 0.1, seed=1)
    by_db = {}
    for e in ex:
        by_db.setdefault(e.db_id, set()).add(e.split)
    assert all(len(s) == 1 for s in by_db.values())
    assert sum(1 for e in ex if e.split == "dev") == 6


def test_check_disjoint_rejects_overlap_and_dupes():
    a = Example("x-1", "train", "db", "q", "s", "f")
    with pytest.raises(SplitError):
        check_disjoint([a, Example("x-2", "test", "db", "q", "s", "f")])
    with pytest.raises(SplitError):
        check_disjoint([a, a])


def test_dev_subset_covers_databases():
    ex = [Example(f"dev-{i:03d}", "dev", f"d{i % 7}", "q", "s", "f") for i in range(100)]
    sub = select_dev_subset(ex, 14, seed=2)
    assert len(sub) == 14 and sub == sorted(sub)
    picked = {e.db_id for e in ex if e.example_id in set(sub)}
    assert len(picked) == 7
    assert sub == select_dev_subset(ex, 14, seed=2)
    assert len(select_dev_subset(ex, 1000, seed=2)) == 100


def test_database_manifest(tmp_path, make_db):
    import shutil

    for name in ("a", "b"):
        (tmp_path / "database" / name).mkdir(parents=True)
        shutil.copy(make_db("CREATE TABLE t (x);"), tmp_path / "database" / name / f"{name}.sqlite")
    m = build_database_manifest(tmp_path / "database", ["b", "a", "a"])
    assert list(m) == ["a", "b"] and len(m["a"]["sha256"]) == 64
    with pytest.raises(FileNotFoundError):
        build_database_manifest(tmp_path / "database", ["a", "zzz"])


def test_supervised_example_masks_prompt_and_keeps_eos():
    tok = FakeTokenizer()
    f = build_supervised_example(tok, "CREATE TABLE t (x);", "How many?", "SELECT count(*) FROM t")
    sup = [i for i, lab in zip(f["input_ids"], f["labels"]) if lab != IGNORE_INDEX]
    first = f["labels"].index(next(lab for lab in f["labels"] if lab != IGNORE_INDEX))
    assert f["labels"][:first] == [IGNORE_INDEX] * first
    assert sup[-1] == tok.eos_token_id
    decoded = {v: k for k, v in tok.vocab.items()}
    assert "".join(decoded[i] for i in sup) == "SELECT count(*) FROM t<|im_end|>"


def test_template_mismatch_detected():
    with pytest.raises(TemplateContractError):
        build_supervised_example(FakeTokenizer(render_eos=False), "s", "q", "SELECT 1")


def test_pad_keeps_eos_supervised_when_pad_equals_eos():
    tok = FakeTokenizer()
    a = build_supervised_example(tok, "s", "q", "SELECT 1")
    b = build_supervised_example(tok, "schema longer here", "q", "SELECT 1 FROM t")
    batch = pad_batch([a, b], tok.pad_token_id)
    la = batch["labels"][0]
    n = len(a["input_ids"])
    assert la[n - 1] == tok.eos_token_id
    assert la[n:] == [IGNORE_INDEX] * (len(la) - n)
    assert batch["attention_mask"][0][n:] == [0] * (len(la) - n)


def test_prompt_fits():
    assert prompt_fits(1792, 2048, 256)
    assert not prompt_fits(1793, 2048, 256)
