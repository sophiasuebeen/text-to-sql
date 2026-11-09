"""Mac-side parts of train.py: config, identity, data filtering. No torch needed."""

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from fakes import FakeTokenizer
from text_to_sql.model.train import ConfigError, load_config, load_training_features, write_status

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
        {"example_id": "train-00001", "split": "train", "db_id": "people", "question": "Bad gold", "query": "SELECT nope FROM people"},
        {"example_id": "train-00002", "split": "train", "db_id": "people", "question": "Names by age", "query": "SELECT name FROM people ORDER BY age"},
        {"example_id": "dev-00000", "split": "dev", "db_id": "people", "question": "Dev q", "query": "SELECT 1"},
        {"example_id": "test-00000", "split": "test", "db_id": "people"},
    ]
    (out / "examples.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    exclusions = [
        {"example_id": "train-00001", "split": "train", "db_id": "people", "reason": "gold_sql_error"},
        {"example_id": "train-00002", "split": "train", "db_id": "people", "reason": "ambiguous_order"},
    ]
    (out / "exclusions.jsonl").write_text("\n".join(json.dumps(r) for r in exclusions))
    return tmp_path


def test_training_features_filter_and_report(tmp_path, phase1):
    cfg = load_config(write_cfg(tmp_path, **{"data.phase1_dir": str(phase1 / "phase1"),
                                             "data.spider_root": str(phase1 / "spider")}))
    features, report = load_training_features(cfg, FakeTokenizer())
    # invalid gold dropped; ambiguous ordering only affects scoring, so it still trains; dev/test never used
    assert (report.candidates, report.dropped_gold_invalid, report.kept) == (3, 1, 2)
    assert all(f["labels"][-1] == FakeTokenizer.eos_token_id for f in features)


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
