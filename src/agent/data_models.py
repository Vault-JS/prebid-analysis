"""
Pydantic data models for LLM agent inputs and outputs.

Defines structured data models for:
- Prebid vendor extraction (PrebidAgent)
- Vendor cluster analysis (VendorAgent)
    - Company, domain, and relationship decisions
    - Evidence and citation tracking

Author: Karel Kubicek <karel.kubicek@vaultjs.com>
"""
from __future__ import annotations
from datetime import datetime
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

class CompanyDecision(BaseModel):
    name: str
    confidence: float | None = None
    rationale: str | None = None
    stage: str | None = None


class DomainDecision(BaseModel):
    domain: str
    relation: Literal["associated_with", "owns", "used_by"] = "associated_with"
    confidence: float | None = None
    rationale: str | None = None


class ParentDecision(BaseModel):
    name: str
    relation: Literal["parent_of", "brand_of", "subsidiary_of", "acquired"] = (
        "parent_of"
    )
    confidence: float | None = None
    rationale: str | None = None


class EvidenceItem(BaseModel):
    source: str
    summary: str
    source_url: str | None = None
    entity_type: Literal["company", "brand", "domain", "link", "relationship"] = (
        "company"
    )
    entity_id: str | None = None
    confidence: float | None = None
    retrieved_at: datetime | None = None


class LinkVendorDecision(BaseModel):
    decision: Literal["link", "new_company", "skip"]
    summary: str | None = None
    company: CompanyDecision | None = None
    domains: list[DomainDecision] = Field(default_factory=list)
    parent: ParentDecision | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    stage: str | None = None


class DomainResource(BaseModel):
    domain: str
    resource_type: Literal["endpoint", "sync", "script", "analytics"]
    url: str
    method: Literal["GET", "POST", "GET/POST"] | None = None
    description: str

    @model_validator(mode="before")
    @classmethod
    def normalize_method(cls, data: Any) -> Any:
        """Normalize method field to handle GET/POST."""
        if isinstance(data, dict) and "method" in data:
            method = data["method"]
            if method == "GET/POST":
                data["method"] = "GET"  # Default to GET for GET/POST
        return data


class ParameterInfo(BaseModel):
    name: str
    scope: Literal["global", "adunit", "rtd"]
    type: str
    default: str | None = None
    description: str

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        """Normalize scope and default fields before validation."""
        if isinstance(data, dict):
            # Handle scope: if it's pipe-separated (e.g., "global|adunit"), take the first one
            if "scope" in data:
                scope = data["scope"]
                if isinstance(scope, str) and "|" in scope:
                    # Take the first scope value
                    data["scope"] = scope.split("|")[0].strip()

            # Normalize default field to always be a string
            if "default" in data:
                default = data["default"]
                if default is not None and not isinstance(default, str):
                    # Convert int, bool, float to string
                    data["default"] = str(default)
        return data


class ConfigurableParameter(BaseModel):
    name: str
    description: str


class PrivacySupport(BaseModel):
    gdpr: bool
    gpp: bool
    ccpa_usp: bool
    coppa: bool
    schain: bool
    eids: bool
    notes: str | None = None


class ExternalScript(BaseModel):
    url: str
    attributes: dict[str, Any] | None = None
    purpose: str


class UserSyncInfo(BaseModel):
    supported: bool
    types: list[Literal["image", "iframe"]] = Field(default_factory=list)
    urls: list[str] | None = None


class ConfigurationExamples(BaseModel):
    global_config: dict[str, Any] | None = None
    adunit_config: dict[str, Any] | None = None
    rtd_config: dict[str, Any] | None = None


class Features(BaseModel):
    floors_supported: bool
    app_supported: bool
    s2s_supported: bool
    user_ids: list[str] = Field(default_factory=list)


class ParametersStructure(BaseModel):
    required: list[ParameterInfo] = Field(default_factory=list)
    optional: list[ParameterInfo] = Field(default_factory=list)
    configurable: list[ConfigurableParameter] = Field(default_factory=list)


class PrebidVendorExtraction(BaseModel):
    vendor_name: str
    product_name: str | None = None
    bidder_code: str | None = None
    maintainer: str | None = None
    domains: list[DomainResource] = Field(default_factory=list)
    parameters: ParametersStructure = Field(default_factory=ParametersStructure)
    privacy_support: PrivacySupport
    features: Features
    gvlid: int | None = None
    supported_media_types: list[Literal["banner", "video", "native"]] = Field(
        default_factory=list
    )
    currency: str | None = None
    external_scripts: list[ExternalScript] = Field(default_factory=list)
    user_sync: UserSyncInfo
    configuration: ConfigurationExamples = Field(default_factory=ConfigurationExamples)
    extraction_notes: str | None = None


class UrlCitation(BaseModel):
    """
    A URL citation extracted from LLM response annotations/grounding.

    Represents a citation to an external URL found in the LLM's response,
    with metadata about where it appears in the text and optional confidence scores.
    """
    url: str = Field(..., description="The URL being cited (resolved/final URL, not redirect).")
    title: str = Field(default="", description="Title of the cited page/document.")
    start_index: int = Field(..., description="Start index in the output text where this citation appears.")
    end_index: int = Field(..., description="End index in the output text where this citation appears.")
    reference_text: str = Field(default="", description="The text excerpt from the output that references this URL.")
    confidence: float | None = Field(None, description="Confidence score from the LLM (if provided). Used when citation is not matched to EvidenceRef.")
    # GPT-specific fields
    gpt_annotation_id: str | None = Field(None, description="GPT annotation ID (if from GPT).")
    # Gemini-specific fields
    gemini_chunk_index: int | None = Field(None, description="Gemini grounding chunk index (if from Gemini).")
    gemini_grounding_support: dict[str, Any] | None = Field(None, description="Full Gemini grounding support data (if from Gemini).")
