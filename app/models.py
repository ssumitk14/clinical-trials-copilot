from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, Union
from datetime import date, datetime
from enum import Enum

# Trial Summary Model
class TrialSummary(BaseModel):
    """Structured summary of a clinical trial"""
    nctid: str
    title: str
    brief_summary: str = ""
    detailed_description: str = ""
    primary_outcome: str = ""
    secondary_outcomes: List[str] = Field(default_factory=list)
    inclusion_criteria: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)
    sponsors: List[str] = Field(default_factory=list)
    phase: List[str] = Field(default_factory=list)


# Basic Trial Information Model
class TrialInfo(BaseModel):
    """Basic clinical trial information"""
    title: str = Field(..., description="Official title of the clinical trial")
    nctid: str = Field(..., description="NCT ID (e.g., NCT12345678)")
    phase: Optional[str] = Field(None, description="Trial phase (Phase I, II, III, IV)")
    status: str = Field(..., description="Current status of the trial")
    participants: Optional[int] = Field(None, description="Number of participants")
    start_date: Optional[date] = Field(None, description="Trial start date")
    completion_date: Optional[date] = Field(None, description="Expected completion date")

class CrossTrials(BaseModel):
    nct_id: str = Field(..., description="Unique identifier for the clinical trial (e.g., NCT01234567).")
    title: Optional[str] = Field(None, description="Brief title of the clinical trial.")
    official_title: Optional[str] = Field(None, description="Official or scientific title.")
    conditions: Optional[List[str]] = Field(None, description="Conditions studied.")
    study_type: Optional[str] = Field(None, description="Type of study (e.g., Interventional, Observational).")
    phase: Optional[str] = Field(None, description="Phase (e.g., Phase 2, Phase 3).")
    status: Optional[str] = Field(None, description="Recruitment status (e.g., Completed, Recruiting).")
    start_date: Optional[str] = Field(None, description="Start date.")
    completion_date: Optional[str] = Field(None, description="Completion date.")
    enrollment_count: Optional[int] = Field(None, description="Enrollment count.")
    sponsor: Optional[str] = Field(None, description="Study sponsor.")
    sex: Optional[str] = Field(None, description="Eligible sex.")
    min_age: Optional[str] = Field(None, description="Minimum age of participants.")
    criteria: Optional[str] = Field(None, description="Inclusion/exclusion criteria.")

    # 👇 Note the change here: allow str OR dict
    interventions: Optional[List[Union[str, dict]]] = Field(None, description="List of interventions.")
    primary_outcomes: Optional[List[Union[str, dict]]] = Field(None, description="Primary outcome measures.")
    secondary_outcomes: Optional[List[Union[str, dict]]] = Field(None, description="Secondary outcome measures.")

    adverse_events: Optional[str] = Field(None, description="Reported adverse events.")

    # flatten dicts into readable text
    @field_validator("interventions", "primary_outcomes", "secondary_outcomes", mode="before")
    @classmethod
    def flatten_dicts(cls, v):
        if not v:
            return v
        if isinstance(v, (dict, str)):
            v = [v]
        flattened = []
        for item in v:
            if isinstance(item, dict):
                flattened.append(", ".join(f"{k}: {v}" for k, v in item.items()))
            else:
                flattened.append(str(item))
        return flattened

    @field_validator(
    "phase", "status", "study_type", "sex", "sponsor", "title", "official_title", mode="before")
    @classmethod
    def flatten_single_item_lists(cls, v):
        """Flatten lists that should be single string fields."""
        if isinstance(v, list):
            return ", ".join(str(i) for i in v)
        return v
