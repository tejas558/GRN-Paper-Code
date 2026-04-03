from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class SampleSpec:
    path: str
    time: Optional[float] = None
    condition: str = "control"
    replicate: str = "rep1"
    sample_name: Optional[str] = None
    kind: str = "single"
    metadata_csv: Optional[str] = None


@dataclass
class PreprocessConfig:
    min_rna_counts: int = 500
    min_atac_counts: int = 500
    min_detected_genes: int = 200
    min_detected_peaks: int = 200
    min_cells_per_gene: int = 10
    min_cells_per_peak: int = 10
    promoter_upstream: int = 2000
    promoter_downstream: int = 500
    n_tfs: int = 100
    n_targets: int = 400
    n_bins: int = 40
    min_cells_per_bin: int = 25
    n_neighbors: int = 20
    rna_pcs: int = 30
    ga_pcs: int = 30


@dataclass
class ModelConfig:
    latent_dim: int = 32
    hidden_dim: int = 128
    edge_rank: int = 8
    max_tau: float = 0.35
    max_step: float = 0.02
    epochs: int = 800
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    seed: int = 13
    recon_weight_tf_acc: float = 1.0
    recon_weight_tf_rna: float = 1.0
    recon_weight_target_rna: float = 1.0
    edge_l1: float = 1e-4
    edge_smooth: float = 1e-4
    latent_l2: float = 1e-4
    deriv_weight: float = 5e-2
    early_stopping_patience: int = 60


@dataclass
class ProjectSpec:
    name: str
    species: str
    gtf: str
    tf_list: Optional[str] = None
    output_dir: str = "outputs"
    samples: List[SampleSpec] = field(default_factory=list)
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


@dataclass
class Config:
    projects: List[ProjectSpec]


_DEF_PRE = PreprocessConfig()
_DEF_MODEL = ModelConfig()


def _coerce_sample(obj: Dict[str, Any]) -> SampleSpec:
    return SampleSpec(**obj)


def _coerce_pre(obj: Optional[Dict[str, Any]]) -> PreprocessConfig:
    if obj is None:
        return PreprocessConfig()
    data = {**_DEF_PRE.__dict__, **obj}
    return PreprocessConfig(**data)


def _coerce_model(obj: Optional[Dict[str, Any]]) -> ModelConfig:
    if obj is None:
        return ModelConfig()
    data = {**_DEF_MODEL.__dict__, **obj}
    return ModelConfig(**data)


def _coerce_project(obj: Dict[str, Any]) -> ProjectSpec:
    samples = [_coerce_sample(x) for x in obj.get("samples", [])]
    preprocessing = _coerce_pre(obj.get("preprocessing"))
    model = _coerce_model(obj.get("model"))
    return ProjectSpec(
        name=obj["name"],
        species=obj["species"],
        gtf=obj["gtf"],
        tf_list=obj.get("tf_list"),
        output_dir=obj.get("output_dir", "outputs"),
        samples=samples,
        preprocessing=preprocessing,
        model=model,
    )


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "projects" not in data:
        raise ValueError("Config file must contain a top-level 'projects' list.")
    projects = [_coerce_project(x) for x in data["projects"]]
    return Config(projects=projects)


def get_project(config: Config, name: str) -> ProjectSpec:
    for project in config.projects:
        if project.name == name:
            return project
    available = ", ".join(p.name for p in config.projects)
    raise KeyError(f"Project '{name}' not found. Available projects: {available}")
