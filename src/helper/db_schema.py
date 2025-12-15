"""
SQLAlchemy ORM schema definitions for the vendor library database.

Uses SQLite for testing, but should be simple to move to Amazon RDS.

Defines all database tables including source data tables (MartechVendors, DdgCompany,
TcfVendor, etc.) and canonical entity tables (Company, BrandOrProduct, Domain,
DomainOrLink, CompanyRelationship, Evidence, EntityCluster, ClusterMember).

Author: Karel Kubicek <karel.kubicek@vaultjs.com>
"""
from datetime import datetime, date
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    MetaData,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


metadata_obj = MetaData(naming_convention=NAMING_CONVENTION)


PROCESSING_STAGE = Enum(
    "raw",
    "llm_linked",
    "verified",
    "reviewed",
    "final",
    name="processing_stage",
    native_enum=False,
    validate_strings=True,
)


class Base(DeclarativeBase):
    metadata = metadata_obj


class PrebidVendor(Base):
    __tablename__ = "prebid_vendors"
    __table_args__ = (
        Index("ix_prebid_vendor_name", "vendor_name"),
        Index("ix_prebid_module_type", "module_type"),
        Index("ix_prebid_stage", "stage"),
        Index("ix_prebid_vendor_module", "vendor_name", "module_type", unique=True),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    vendor_name: Mapped[str] = mapped_column(Text, nullable=False)
    module_type: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # BidAdapter, RtdProvider, AnalyticsAdapter, IdSystem, VideoProvider
    js_file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    md_file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    ts_file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    js_file_hash: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # SHA256 hash of js file
    md_file_hash: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # SHA256 hash of md file
    ts_file_hash: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # SHA256 hash of ts file
    raw_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )  # Store file sizes, line counts, etc.
    extracted_data: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )  # Will be populated by LLM
    stage: Mapped[str] = mapped_column(PROCESSING_STAGE, default="raw", nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )
    used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class PrebidVendorExtraction(Base):
    __tablename__ = "prebid_vendor_extractions"
    __table_args__ = (
        Index("ix_prebid_extraction_vendor", "prebid_vendor_id"),
        Index("ix_prebid_extraction_stage", "stage"),
        Index("ix_prebid_extraction_llm_request", "llm_request_id"),
        Index("ix_prebid_extraction_extracted_at", "extracted_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    prebid_vendor_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("prebid_vendors.id"), nullable=False
    )
    llm_request_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("llm_requests.id"), nullable=True
    )

    # Core extracted fields (from prompt output)
    vendor_name: Mapped[str] = mapped_column(Text, nullable=False)
    product_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    bidder_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    maintainer: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Privacy support flags
    gdpr_supported: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    gpp_supported: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    ccpa_usp_supported: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    coppa_supported: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    schain_supported: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    eids_supported: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Advanced Features
    floors_supported: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    app_supported: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    s2s_supported: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    user_ids: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    # Metadata
    gvlid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    supported_media_types: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    currency: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Full extracted JSON (for complex nested structures)
    extracted_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Quality/confidence tracking
    extraction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    extraction_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    stage: Mapped[str] = mapped_column(PROCESSING_STAGE, default="raw", nullable=False)
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )


class PrebidDoc(Base):
    __tablename__ = "prebid_docs"
    __table_args__ = (
        Index("ix_prebid_doc_vendor", "prebid_vendor_id"),
        Index("ix_prebid_doc_slug", "slug", unique=True),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    prebid_vendor_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("prebid_vendors.id"), nullable=True
    )
    slug: Mapped[str] = mapped_column(Text, nullable=False)  # filename stem
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    raw_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Metadata for tracking updates
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )


class DeviceStorageDisclosure(Base):
    __tablename__ = "device_storage_disclosures"
    __table_args__ = (
        Index("ix_device_storage_tcf_vendor", "tcf_2v2_vendor_id"),
        Index("ix_device_storage_collection_date", "collection_date"),
        Index("ix_device_storage_disclosure_url", "disclosure_url"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tcf_2v2_vendor_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("tcf_vendors.tcf_2v2_vendor_id"), nullable=True, unique=True
    )
    disclosure_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    prebid_vendor_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_response: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    disclosures: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON, nullable=True
    )
    domains: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    collection_date: Mapped[date] = mapped_column(Date, nullable=False)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )


class TcfVendor(Base):
    __tablename__ = "tcf_vendors"
    __table_args__ = (
        Index("ix_tcf_vendor_name", "vendor_name"),
        Index("ix_tcf_iab_vendor_id", "iab_europe_vendor_id"),
        Index("ix_tcf_v1_vendor_id", "tcf_v1_vendor_id"),
        Index("ix_tcf_2v2_vendor_id", "tcf_2v2_vendor_id"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    vendor_name: Mapped[str] = mapped_column(Text, nullable=False)
    iab_europe_vendor_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True, unique=True
    )
    tcf_v1_vendor_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True, unique=True
    )
    tcf_2v2_vendor_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True, unique=True
    )
    iab_europe_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    tcf_v1_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    tcf_2v2_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    collection_date: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )
    used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


# Helper tables for processing


class LlmRequest(Base):
    __tablename__ = "llm_requests"
    __table_args__ = (
        Index("ix_llm_request_response_id", "response_id"),
        Index("ix_llm_request_batch_id", "batch_id"),
        Index("ix_llm_request_status", "status"),
        Index("ix_llm_request_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    batch_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )  # Groups related batch requests
    response_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, unique=True
    )  # OpenAI response ID
    model: Mapped[str] = mapped_column(Text, nullable=False)
    service_tier: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # pending, completed, failed
    context: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    request_data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    response_data: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )  # Full response
    cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)



def create_all_tables(engine) -> None:
    Base.metadata.create_all(engine)
