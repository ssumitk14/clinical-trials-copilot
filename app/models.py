from pydantic import BaseModel, Field
from typing import List, Optional, Literal
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

