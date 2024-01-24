import dataclasses
import os

from typing import List, Optional


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    num_nq: int
    """Number of NQ queries to use if maximize_queries is False, otherwise the number of documents to use"""
    val_pct: int
    """Percent of NQ queries to hold out for validation (0-100)"""
    force_train_docs_in_val: bool
    """Force all documents in validation to appear in train"""
    prepend_title: bool
    """Prepend the title to the document"""
    maximize_queries: bool
    """Maximize the number of queries for a given index size (num_nq)"""


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    max_doc_length: int
    ratio_indexing_to_query_train: float
    ratio_indexing_to_query_val: float
    batch_size: int
    num_steps: int
    eval_steps: int
    learning_rate: float
    save_steps: int
    label_length: int
    num_eval_queries: int
    batch_size_eval: int
    sample_doc_chunks_train: bool
    sample_doc_chunks_val: bool


@dataclasses.dataclass(frozen=True)
class _ExperimentID:
    name: str
    """Name of the experiment (also the name of the directory), passed as group to wandb"""
    run: str
    """Human identifier for the run (also the name of the subdirectory), passed as name to wandb"""


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    """User's should initialize this below in _CONFIGS"""

    eid: _ExperimentID
    seed: int
    base_model_name: str
    data: DatasetConfig
    train: TrainConfig
    notes: str = ""
    base_dir: str = ""
    """Base directory for this experiment, all data will be stored in subdirectories of this directory.
    NOTE: this should not be set as part of the config file, instead
        users should call get_experiment_config and pass this along,
        this is to avoid hardcoding the base_dir in this file.
    """
    base_eid: Optional[_ExperimentID] = None
    """If this is a sub-experiment, this should be the ExperimentID of the parent experiment"""

    def _path(self, *parts) -> str:
        return os.path.join(self.base_dir, self.eid.name, self.eid.run, *parts)

    def dataset_dir(self) -> str:
        return self._path("dataset")

    def model_dir(self) -> str:
        return self._path("model")

    def working_dir(self) -> str:
        return self._path()


@dataclasses.dataclass(frozen=True)
class ExperimentExtensionConfig:
    eid: _ExperimentID
    parent: _ExperimentID
    seed: int

    def config(self, parent: ExperimentConfig) -> ExperimentConfig:
        return dataclasses.replace(
            parent, eid=self.eid, seed=self.seed, base_eid=parent
        )


_CONFIGS: List[ExperimentConfig] = [
    ExperimentConfig(
        eid=_ExperimentID(
            name="experiment_sample_doc_chunks",
            run="sanity_check",
        ),
        seed=42,
        base_model_name="t5-small",
        data=DatasetConfig(
            num_nq=125,
            val_pct=20,
            force_train_docs_in_val=False,
            prepend_title=False,
            maximize_queries=False,
        ),
        train=TrainConfig(
            max_doc_length=32,
            ratio_indexing_to_query_train=32,
            ratio_indexing_to_query_val=1,
            batch_size=512,
            num_steps=10000,
            eval_steps=100,
            learning_rate=4e-5,
            save_steps=1000,
            label_length=8,
            num_eval_queries=256,
            batch_size_eval=128,
            sample_doc_chunks_train=True,
            sample_doc_chunks_val=True,
        ),
        notes="Sanity check to make sure everything is working, sample doc chunks for doc representation for indexing task in train and val",
    ),
    ExperimentConfig(
        eid=_ExperimentID(
            name="test_save",
            run="sanity_check",
        ),
        seed=42,
        base_model_name="t5-small",
        data=DatasetConfig(
            num_nq=125,
            val_pct=20,
            force_train_docs_in_val=False,
            prepend_title=False,
            maximize_queries=False,
        ),
        train=TrainConfig(
            max_doc_length=32,
            ratio_indexing_to_query_train=32,
            ratio_indexing_to_query_val=1,
            batch_size=512,
            num_steps=10,
            eval_steps=100,
            learning_rate=4e-5,
            save_steps=1000,
            label_length=8,
            num_eval_queries=256,
            batch_size_eval=128,
            sample_doc_chunks_train=True,
            sample_doc_chunks_val=True,
        ),
        notes="Sanity check to make sure everything is working, sample doc chunks for doc representation for indexing task in train and val",
    ),
]


_EXTENSIONS: List[ExperimentExtensionConfig] = [
    ExperimentExtensionConfig(
        eid=_ExperimentID(
            name="test_save_extension",
            run="sanity_check",
        ),
        parent=_ExperimentID(
            name="test_save",
            run="sanity_check",
        ),
        seed=42,
    ),
]

assert len(_CONFIGS) + len(_EXTENSIONS) == len(
    set([c.eid for c in _CONFIGS]).union(set([c.eid for c in _EXTENSIONS]))
), "Duplicate experiment IDs"


def get_experiment_config(name: str, run: str, base_dir: str) -> ExperimentConfig:
    eid = _ExperimentID(name=name, run=run)
    matches = [c for c in _CONFIGS if c.eid == eid]
    if len(matches) == 1:
        return dataclasses.replace(matches[0], base_dir=base_dir)

    matches = [c for c in _EXTENSIONS if c.eid == eid]
    assert len(matches) == 1, f"Expected 1 match, got {len(matches)} for {eid}"
    child = matches[0]

    parents = [c for c in _CONFIGS if c.eid == child.parent]
    assert len(parents) == 1, f"Expected 1 parent, got {len(parents)}"
    parent = parents[0]
    parent = dataclasses.replace(parent, base_dir=base_dir)
    return child.config(parent)
