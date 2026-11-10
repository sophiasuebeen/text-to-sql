import json
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from text_to_sql import EVALUATOR_VERSION
from text_to_sql.data.schema import SERIALIZER_VERSION
from text_to_sql.evaluation.compare_dev import compare_dev, select_examples
from text_to_sql.evaluation.reference import ReferenceCache
from text_to_sql.io import sha256_file, write_jsonl
from text_to_sql.model.checkpoints import publish_checkpoint
from text_to_sql.model.predict import Prediction, SQLPredictor, export_identity
from text_to_sql.model.prompts import PROMPT_VERSION


def test_export_requires_verified_matching_identity(tmp_path):
    identity = {"model": "example/model", "model_revision": "a" * 40,
                "prompt_version": PROMPT_VERSION, "serializer_version": SERIALIZER_VERSION}
    local = tmp_path / "local" / "final"
    local.mkdir(parents=True)
    (local / "identity.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="complete"):
        export_identity(local)
    saved = publish_checkpoint(local, tmp_path / "drive", identity)
    assert export_identity(saved) == identity
    (saved / "identity.json").write_text("{}")
    with pytest.raises(RuntimeError):
        export_identity(saved)


@pytest.fixture
def diagnostic(tmp_path, people_db, fast_limits):
    phase1 = tmp_path / "phase1"
    phase1.mkdir()
    pins = {"bundle": {"url": "fixture", "sha256": "b" * 64},
            "hf_dataset": {"name": "fixture", "revision": "c" * 40}}
    cfg = {"output_dir": str(phase1), **pins}
    # Reuse the fixture DB without changing its location.
    root = tmp_path / "spider" / "database" / "people"
    root.mkdir(parents=True)
    (root / "people.sqlite").write_bytes(people_db.read_bytes())
    cfg["spider_root"] = str(tmp_path / "spider")
    config = tmp_path / "data.yaml"
    config.write_text(yaml.safe_dump(cfg))
    rows = [{"example_id": f"dev-{i}", "split": "dev", "db_id": "people",
             "question": f"question {i}", "query": "SELECT count(*) FROM people"} for i in range(3)]
    rows += [{"example_id": "test-0", "split": "test", "db_id": "people"}]
    write_jsonl(phase1 / "examples.jsonl", rows)
    write_jsonl(phase1 / "exclusions.jsonl", [{"example_id": "dev-2", "reason": "needs_adjudication"}])
    (phase1 / "dev_subset.json").write_text(json.dumps(["test-0", "dev-2", "dev-1", "dev-0"]))
    (phase1 / "integrity_summary.json").write_text(json.dumps({
        "comparator_defects": [], "unpinned_dry_run": False, "pins": pins,
        "evaluator_version": EVALUATOR_VERSION, "limits": asdict(fast_limits),
        "pending_adjudication": 1,
    }))
    digest = sha256_file(people_db)
    (phase1 / "databases.json").write_text(json.dumps({"people": {"sha256": digest}}))
    ReferenceCache(phase1 / "reference_cache.jsonl").get_or_audit(
        people_db, digest, rows[0]["query"], fast_limits,
    )
    return config, phase1


def test_selection_keeps_only_frozen_eligible_dev(diagnostic):
    _, phase1 = diagnostic
    examples, _ = select_examples(phase1)
    assert [r["example_id"] for r in examples] == ["dev-0", "dev-1"]
    assert len(select_examples(phase1, 1)[0]) == 1
    with pytest.raises(ValueError):
        select_examples(phase1, 0)


def test_paired_diagnostic_scores_real_sql_and_keeps_generation_failure_in_denominator(
    diagnostic, tmp_path, monkeypatch,
):
    import text_to_sql.evaluation.compare_dev as module

    config, _ = diagnostic
    calls = []

    class Predictor:
        def __init__(self, *args, **kwargs):
            self.identity = {"model": "fixture"}

        def predict(self, schema, question, *, baseline=False):
            calls.append((schema, question, baseline))
            if baseline and question == "question 1":
                return Prediction("", None, "prompt_overflow", 9999, 0)
            sql = "SELECT count(*) FROM people" if baseline or question == "question 1" else "SELECT 'Brown"
            return Prediction(sql, sql, "ok", 10, 0)

    monkeypatch.setattr(module, "SQLPredictor", Predictor)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda: 0)))
    output = tmp_path / "results"
    result = compare_dev(adapter=Path("unused"), data_config=config, output=output)
    assert result["n_examples"] == 2
    assert result["baseline_accuracy"] == result["adapter_accuracy"] == 0.5
    assert [(q, b) for _, q, b in calls] == [
        ("question 0", True), ("question 0", False), ("question 1", True), ("question 1", False),
    ]
    assert calls[0][0] == calls[1][0]
    saved = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert saved[1]["baseline"]["prediction"]["status"] == "prompt_overflow"
    assert not saved[1]["baseline"]["score"]["correct"]
    assert saved[0]["adapter"]["score"]["pred_status"] == "parse_error"
    assert result["provisional_dev_diagnostic"]
    with pytest.raises(FileExistsError):
        compare_dev(adapter=Path("unused"), data_config=config, output=output)

    # Simulate interruption after the first complete pair. Reuse its generation,
    # but recompute its scores, and generate only the remaining example.
    saved[0]["adapter"]["score"]["correct"] = True
    write_jsonl(output / "predictions.jsonl", saved[:1])
    (output / "summary.json").unlink()
    calls.clear()
    resumed = compare_dev(adapter=Path("unused"), data_config=config, output=output, resume=True)
    assert resumed["resumed_examples"] == 1
    assert resumed["baseline_accuracy"] == resumed["adapter_accuracy"] == 0.5
    assert [(q, b) for _, q, b in calls] == [("question 1", True), ("question 1", False)]
    assert len((output / "predictions.jsonl").read_text().splitlines()) == 2

    before = (output / "predictions.jsonl").read_bytes()
    calls.clear()
    with pytest.raises(ValueError, match="settings changed"):
        compare_dev(adapter=Path("unused"), data_config=config, output=output, resume=True, max_new_tokens=128)
    assert not calls and (output / "predictions.jsonl").read_bytes() == before

    # A duplicate/non-prefix ID must not silently inflate the denominator.
    rows = [json.loads(line) for line in before.decode().splitlines()]
    rows[1] = rows[0]
    write_jsonl(output / "predictions.jsonl", rows)
    with pytest.raises(ValueError, match="ordered prefix"):
        compare_dev(adapter=Path("unused"), data_config=config, output=output, resume=True)


def test_changed_database_rejected_before_loading_model(diagnostic, tmp_path):
    config, phase1 = diagnostic
    (phase1 / "databases.json").write_text(json.dumps({"people": {"sha256": "0" * 64}}))
    with pytest.raises(ValueError, match="database changed"):
        compare_dev(adapter=Path("unused"), data_config=config, output=tmp_path / "results")


@pytest.mark.parametrize("overflow,truncated", [(False, False), (True, False), (False, True)])
def test_predict_switches_adapter_restores_it_and_rejects_incomplete_output(monkeypatch, overflow, truncated):
    from fakes import FakeTokenizer

    calls = []

    class Encoded(dict):
        def to(self, device):
            return self

    class Tokenizer(FakeTokenizer):
        def __call__(self, text, **kwargs):
            return Encoded(input_ids=SimpleNamespace(shape=(1, 9999 if overflow else 10)))

        def decode(self, tokens, **kwargs):
            return "SELECT 1"

    class Model:
        device = "cuda:0"
        disabled = False

        @contextmanager
        def disable_adapter(self):
            self.disabled = True
            try:
                yield
            finally:
                self.disabled = False

        def generate(self, **kwargs):
            calls.append((self.disabled, kwargs))
            tokens = [100] * 16 if truncated else [100, 2]

            class Output:
                def __getitem__(self, key):
                    assert key == (0, slice(10, None))
                    return SimpleNamespace(tolist=lambda: tokens)

            return Output()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        inference_mode=nullcontext, cuda=SimpleNamespace(synchronize=lambda: None),
    ))
    predictor = SQLPredictor.__new__(SQLPredictor)
    predictor.model = Model()
    predictor.tokenizer = Tokenizer()
    predictor.max_context = 100
    predictor.max_new_tokens = 16
    first = predictor.predict('CREATE TABLE "people" ("id" INTEGER);', "Count people", baseline=True)
    second = predictor.predict('CREATE TABLE "people" ("id" INTEGER);', "Count people")
    assert not predictor.model.disabled
    if overflow:
        assert first.status == second.status == "prompt_overflow"
        assert not calls
    else:
        assert [disabled for disabled, _ in calls] == [True, False]
        assert all(not kwargs["do_sample"] and kwargs["num_beams"] == 1 for _, kwargs in calls)
        assert first.sql == second.sql == (None if truncated else "SELECT 1")
        assert first.status == ("generation_limit" if truncated else "ok")
