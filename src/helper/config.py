"""
Application configuration management using Pydantic models.

Loads and validates configuration from YAML files, including ingestion settings,
LLM configuration, database credentials (with AWS Secrets Manager support),
and tool thresholds.

Author: Karel Kubicek <karel.kubicek@vaultjs.com>
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal, Any

from pydantic import BaseModel, Field, field_validator
import yaml


class IngestionConfig(BaseModel):
    canonical_db_path: Path
    martech_file_paths: list[Path] = Field(min_length=1)
    tracker_radar_repo_url: str
    tracker_radar_temp_dir: Path
    tracker_radar_extract_dirname: str
    tracker_radar_download_timeout_seconds: int = Field(gt=0)
    tcf_iab_europe_url: str
    tcf_iab_europe_ajax_url: str
    tcf_onetrust_url: str
    tcf_iab2v2_url: str
    tcf_google_url: str
    tcf_browser_timeout_seconds: int = Field(gt=0, default=60)
    builtwith_base_url: str
    builtwith_browser_timeout_seconds: int = Field(gt=0, default=30)
    builtwith_page_load_wait_ms: int = Field(gt=0, default=5000)
    prebid_repo_url: str
    prebid_temp_dir: Path
    prebid_extract_dirname: str
    prebid_download_timeout_seconds: int = Field(gt=0, default=120)
    gtm_api_url: str
    gtm_timeout_seconds: int = Field(gt=0, default=30)
    gtm_raw_output_path: Path
    disconnect_repo_url: str
    disconnect_temp_dir: Path
    disconnect_extract_dirname: str
    disconnect_download_timeout_seconds: int = Field(gt=0, default=120)

    @property
    def tracker_radar_extract_path(self) -> Path:
        return self.tracker_radar_temp_dir / self.tracker_radar_extract_dirname

    @field_validator("martech_file_paths")
    @classmethod
    def ensure_unique_paths(cls, value: list[Path]) -> list[Path]:
        seen: set[Path] = set()
        duplicates: set[Path] = set()
        for path in value:
            resolved = Path(path)
            if resolved in seen:
                duplicates.add(resolved)
            seen.add(resolved)
        if duplicates:
            duplicate_list = ", ".join(str(dup) for dup in sorted(duplicates))
            raise ValueError(f"Duplicate martech_file_paths entries: {duplicate_list}")
        return value


class AnalysisConfig(BaseModel):
    plot_output_dir: Path


class PrebidExtractionConfig(BaseModel):
    provider: Literal["openai", "gemini"] = "openai"
    model_name: str
    reasoning_effort: str
    enable_web_search: bool = False
    search_context_size: Literal["medium", "high"] = "medium"
    service_tier: Literal["flex", "default", "priority"] = "flex"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for Gemini models (0.0-2.0). Not used for GPT thinking models.")


class LlmConfig(BaseModel):
    prebid_extraction: PrebidExtractionConfig


class ToolsConfig(BaseModel):
    thresholds: dict[str, int] = Field(default_factory=dict)


class AppConfig(BaseModel):
    ingestion: IngestionConfig
    analysis: AnalysisConfig
    llm: LlmConfig
    tools: ToolsConfig | None = None

    def resolve_paths(self, base_path: Path) -> "AppConfig":
        def to_absolute(path: Path) -> Path:
            return path if path.is_absolute() else (base_path / path).resolve()

        ingestion = self.ingestion.model_copy(
            update={
                "canonical_db_path": to_absolute(self.ingestion.canonical_db_path),
                "martech_file_paths": [
                    to_absolute(path) for path in self.ingestion.martech_file_paths
                ],
                "tracker_radar_temp_dir": to_absolute(
                    self.ingestion.tracker_radar_temp_dir
                ),
                "prebid_temp_dir": to_absolute(self.ingestion.prebid_temp_dir),
                "gtm_raw_output_path": to_absolute(self.ingestion.gtm_raw_output_path),
                "disconnect_temp_dir": to_absolute(self.ingestion.disconnect_temp_dir),
            }
        )

        analysis = self.analysis.model_copy(
            update={"plot_output_dir": to_absolute(self.analysis.plot_output_dir)}
        )

        return self.model_copy(
            update={
                "ingestion": ingestion,
                "analysis": analysis,
            }
        )


def load_app_config(path: Path | str | None = None) -> AppConfig:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config.yaml"

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config: Any = yaml.safe_load(handle)

    config = AppConfig.model_validate(raw_config)

    project_root = config_path.parent
    return config.resolve_paths(project_root)
