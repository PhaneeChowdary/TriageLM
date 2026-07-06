"""Output schema for a triaged ticket."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Priority(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class Entities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str | None = None
    account_type: str | None = None
    account_category: str | None = None
    person_name: str | None = None
    refund_amount: str | None = None
    delivery_city: str | None = None
    delivery_country: str | None = None
    invoice_id: str | None = None


class TriageOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str
    category: str
    entities: Entities = Field(default_factory=Entities)
    priority: Priority        # derived
    requires_human: bool      # derived
    suggested_queue: str      # derived


def validate_output(raw: dict) -> TriageOutput:
    return TriageOutput.model_validate(raw)
