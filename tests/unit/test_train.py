"""Mac-side parts of train.py: config, identity, data filtering. No torch needed."""

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from fakes import FakeTokenizer
from text_to_sql.model.train import ConfigError, load_config, load_training_features, prepare_training_data, write_status

REPO = Path(__file__).resolve().parents[2]


def write_cfg(tmp_path, **overrides):
    raw = yaml.safe_load((REPO / "configs" / "train.yaml").read_text())
    raw["model"]["revision"] = "deadbeef"
    for dotted, value in overrides.items():
        node = raw
        *path, last = dotted.split(".")
        for k in path:
            node = node[k]
        node[last] = value
    p = tmp_path / "train.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_repo_config_loads_with_plan_values(tmp_path):
    cfg = load_config(write_cfg(tmp_path))
    assert (cfg.lora.r, cfg.lora.alpha, cfg.lora.dropout) == (16, 32, 0.05)
    assert cfg.lora.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert cfg.training.effective_batch_size == 16
    assert cfg.training.learning_rate == 2e-4


def test_unpinned_model_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write_cfg(tmp_path, **{"model.revision": None}))
    assert load_config(write_cfg(tmp_path, **{"model.revision": None}), allow_unpinned=True)


def test_effective_batch_must_stay_16(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write_cfg(tmp_path, **{"training.per_device_train_batch_size": 2}))
    ok = load_config(write_cfg(tmp_path, **{"training.per_device_train_batch_size": 2,
                                            "training.gradient_accumulation_steps": 8}))
    assert ok.training.effective_batch_size == 16


def test_smoke_overrides_and_separate_identity(tmp_path):
    p = write_cfg(tmp_path)
    full, smoke = load_config(p), load_config(p, smoke=True)
    assert smoke.run_name == full.run_name + "-smoke"
    assert smoke.max_train_examples == 200 and smoke.training.save_steps == 10
    assert smoke.training.learning_rate == full.training.learning_rate
    assert smoke.semantic_hash() != full.semantic_hash()


def test_semantic_hash_ignores_paths(tmp_path):
    a = load_config(write_cfg(tmp_path))
    b = load_config(write_cfg(tmp_path, **{"output.drive_dir": "/elsewhere"}))
    c = load_config(write_cfg(tmp_path, **{"lora.r": 8}))
    assert a.semantic_hash() == b.semantic_hash() != c.semantic_hash()


def test_train_module_does_not_import_ml_libraries():
    assert not {"torch", "transformers", "peft", "bitsandbytes"} & set(sys.modules)


@pytest.fixture
def phase1(tmp_path, people_db):
    db_dir = tmp_path / "spider" / "database" / "people"
    db_dir.mkdir(parents=True)
    shutil.copy(people_db, db_dir / "people.sqlite")
    out = tmp_path / "phase1"
    out.mkdir()
    rows = [
        {"example_id": "train-00000", "split": "train", "db_id": "people", "question": "How many?", "query": "SELECT count(*) FROM people"},
        {"example_id": "train-00001", "split": "train", "db_id": "people", "question": "Bad reference", "query": "SELECT nope FROM people"},
        {"example_id": "train-00002", "split": "train", "db_id": "people", "question": "Names by age", "query": "SELECT name FROM people ORDER BY age"},
        {"example_id": "dev-00000", "split": "dev", "db_id": "people", "question": "Dev q", "query": "SELECT 1"},
        {"example_id": "test-00000", "split": "test", "db_id": "people"},
    ]
    (out / "examples.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    exclusions = [
        {"example_id": "train-00001", "split": "train", "db_id": "people", "reason": "reference_sql_error"},
        {"example_id": "train-00002", "split": "train", "db_id": "people", "reason": "ambiguous_order"},
    ]
    (out / "exclusions.jsonl").write_text("\n".join(json.dumps(r) for r in exclusions))
    return tmp_path


@pytest.mark.parametrize("reason", ["reference_sql_error", "gold_sql_error"])
def test_training_features_filter_and_report(tmp_path, phase1, reason):
    exclusions = phase1 / "phase1" / "exclusions.jsonl"
    exclusions.write_text(exclusions.read_text().replace("reference_sql_error", reason))
    cfg = load_config(write_cfg(tmp_path, **{"data.phase1_dir": str(phase1 / "phase1"),
                                             "data.spider_root": str(phase1 / "spider")}))
    features, report = load_training_features(cfg, FakeTokenizer())
    # invalid reference dropped; ambiguous ordering only affects scoring, so it still trains; dev/test never used
    assert (report.candidates, report.dropped_reference_invalid, report.kept) == (3, 1, 2)
    assert all(f["labels"][-1] == FakeTokenizer.eos_token_id for f in features)


def test_training_diagnostic_reconstructs_rows_then_applies_scoring_exclusions(tmp_path, phase1, monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace

    import text_to_sql.evaluation.compare_dev as evaluation

    config = write_cfg(tmp_path, **{
        "data.phase1_dir": str(phase1 / "phase1"), "data.spider_root": str(phase1 / "spider"),
    })
    cfg = load_config(config, smoke=True)
    features, report, trained = prepare_training_data(cfg, FakeTokenizer())
    assert [r["example_id"] for r in trained] == ["train-00000", "train-00002"]
    assert len(features) == report.kept == len(trained)
    adapter = tmp_path / "run" / "final"
    adapter.mkdir(parents=True)
    report_path = adapter.parent / "data_report.json"
    report_path.write_text(json.dumps(asdict(report)))
    identity = {"config_hash": cfg.semantic_hash(), "model": cfg.model_name, "model_revision": cfg.model_revision}
    monkeypatch.setattr(evaluation, "export_identity", lambda path: identity)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda path: FakeTokenizer()),
    ))
    kwargs = dict(adapter=adapter, training_config=config,
                  phase1=phase1 / "phase1", spider_root=phase1 / "spider")
    examples, _exclusions, info = evaluation.select_training_examples(**kwargs)
    assert [r["example_id"] for r in examples] == ["train-00000"]
    assert info["n_trained_examples"] == 2
    assert info["n_training_examples_excluded_from_scoring"] == 1
    assert info["trained_example_ids"] == [r["example_id"] for r in trained]

    report_path.write_text(json.dumps({**asdict(report), "kept": 999}))
    with pytest.raises(ValueError, match="data report differs"):
        evaluation.select_training_examples(**kwargs)
    identity["config_hash"] = "different run"
    with pytest.raises(ValueError, match="config hash"):
        evaluation.select_training_examples(**kwargs)


def test_overlength_dropped_not_truncated(tmp_path, phase1):
    cfg = load_config(write_cfg(tmp_path, **{"data.phase1_dir": str(phase1 / "phase1"),
                                             "data.spider_root": str(phase1 / "spider"),
                                             "context.max_length": 10}))
    features, report = load_training_features(cfg, FakeTokenizer())
    assert features == [] and report.dropped_overlength == 2


def test_smoke_subset_is_seeded(tmp_path, phase1):
    p = write_cfg(tmp_path, **{"data.phase1_dir": str(phase1 / "phase1"), "data.spider_root": str(phase1 / "spider"),
                               "data.max_train_examples": 2})
    cfg = load_config(p)
    a, ra = load_training_features(cfg, FakeTokenizer())
    b, rb = load_training_features(cfg, FakeTokenizer())
    assert ra.candidates == 2 and [f["input_ids"] for f in a] == [f["input_ids"] for f in b]


def test_status_is_written_atomically(tmp_path):
    write_status(tmp_path / "s" / "status.json", run="r", state="training", step=3, loss=1.5)
    d = json.loads((tmp_path / "s" / "status.json").read_text())
    assert d["state"] == "training" and d["step"] == 3 and "heartbeat_utc" in d
    assert not list((tmp_path / "s").glob("*.tmp"))
