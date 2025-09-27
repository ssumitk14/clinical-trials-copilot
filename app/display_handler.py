import streamlit as st
import pandas as pd
from typing import Any

class TrialDisplayHandler:
    """Handler class for displaying Pydantic trial objects in Streamlit"""
    
    @staticmethod
    def display_basic_format(trial_data: Any) -> None:
        """
        Basic formatted display with sections
        """
        st.subheader("📋 Trial Information")
        
        # Basic Information Section
        st.markdown("### Basic Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**NCT ID:** {trial_data.nctid}")
            st.write(f"**Title:** {trial_data.title}")
        
        with col2:
            if hasattr(trial_data, 'primary_outcome') and trial_data.primary_outcome:
                st.write(f"**Primary Outcome:** {trial_data.primary_outcome}")
        
        # Summary Section
        if hasattr(trial_data, 'brief_summary') and trial_data.brief_summary:
            st.markdown("### Summary")
            st.write(trial_data.brief_summary)
        
        # Detailed Description
        if hasattr(trial_data, 'detailed_description') and trial_data.detailed_description:
            st.markdown("### Detailed Description") 
            st.write(trial_data.detailed_description)
        
        # Secondary Outcomes
        if hasattr(trial_data, 'secondary_outcomes') and trial_data.secondary_outcomes:
            st.markdown("### Secondary Outcomes")
            for i, outcome in enumerate(trial_data.secondary_outcomes, 1):
                st.write(f"{i}. {outcome}")
        
        # Criteria sections
        TrialDisplayHandler._display_criteria_sections(trial_data)
        
        # Sponsors
        if hasattr(trial_data, 'sponsors') and trial_data.sponsors:
            st.markdown("### Sponsors")
            for sponsor in trial_data.sponsors:
                st.write(f"• {sponsor}")

    @staticmethod
    def display_card_format(trial_data: Any) -> None:
        """
        Card-style display with metrics and expandable sections
        """
        # Header with trial ID
        st.title(f"🏥 {trial_data.nctid}")
        
        # Title in a nice box
        st.info(f"**{trial_data.title}**")
        
        # Key metrics
        TrialDisplayHandler._display_metrics(trial_data)
        
        # Main content in expandable sections
        with st.expander("📖 Summary", expanded=True):
            if hasattr(trial_data, 'brief_summary') and trial_data.brief_summary:
                st.write(trial_data.brief_summary)
            else:
                st.write("No summary available")
        
        if hasattr(trial_data, 'detailed_description') and trial_data.detailed_description:
            with st.expander("📝 Detailed Description"):
                st.write(trial_data.detailed_description)
        
        # Primary Outcome
        if hasattr(trial_data, 'primary_outcome') and trial_data.primary_outcome:
            with st.expander("🎯 Primary Outcome"):
                st.write(trial_data.primary_outcome)
        
        # Secondary Outcomes
        if hasattr(trial_data, 'secondary_outcomes') and trial_data.secondary_outcomes:
            with st.expander("📊 Secondary Outcomes"):
                for i, outcome in enumerate(trial_data.secondary_outcomes, 1):
                    st.write(f"**{i}.** {outcome}")
        
        # Eligibility Criteria
        TrialDisplayHandler._display_criteria_expandable(trial_data)
        
        # Sponsors
        if hasattr(trial_data, 'sponsors') and trial_data.sponsors:
            with st.expander("🏢 Sponsors"):
                for sponsor in trial_data.sponsors:
                    st.write(f"• {sponsor}")

    @staticmethod
    def display_tab_format(trial_data: Any) -> None:
        """
        Tab-based layout for organized information display
        """
        # Main title
        st.title(trial_data.title)
        st.caption(f"NCT ID: {trial_data.nctid}")
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Outcomes", "Eligibility", "Raw Data"])
        
        with tab1:
            TrialDisplayHandler._display_overview_tab(trial_data)
        
        with tab2:
            TrialDisplayHandler._display_outcomes_tab(trial_data)
        
        with tab3:
            TrialDisplayHandler._display_eligibility_tab(trial_data)
        
        with tab4:
            TrialDisplayHandler._display_raw_data_tab(trial_data)

    @staticmethod
    def display_dataframe_format(trial_data: Any) -> None:
        """
        Display with emphasis on dataframes for structured data
        """
        st.title(trial_data.title)
        st.markdown(f"**NCT ID:** {trial_data.nctid}")
        
        # Basic info
        if hasattr(trial_data, 'brief_summary') and trial_data.brief_summary:
            st.write("**Summary:**", trial_data.brief_summary)
        
        # Convert secondary outcomes to DataFrame
        if hasattr(trial_data, 'secondary_outcomes') and trial_data.secondary_outcomes:
            st.subheader("Secondary Outcomes")
            outcomes_df = pd.DataFrame({
                'Outcome': trial_data.secondary_outcomes
            })
            st.dataframe(outcomes_df, use_container_width=True, hide_index=True)
        
        # Show all available data in a structured way
        st.subheader("Complete Trial Data")
        summary_df = TrialDisplayHandler._create_summary_dataframe(trial_data)
        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    @staticmethod
    def display_comprehensive_format(trial_data: Any) -> None:
        """
        Comprehensive display with all features including missing data handling
        """
        # Header section
        st.markdown(f"## {trial_data.title}")
        st.markdown(f"**NCT ID:** `{trial_data.nctid}`")
        
        # Metrics row
        TrialDisplayHandler._display_metrics(trial_data)
        
        # Summary card
        if hasattr(trial_data, 'brief_summary') and trial_data.brief_summary:
            st.markdown("---")
            st.markdown("### 📄 Summary")
            st.info(trial_data.brief_summary)
        
        # Primary outcome highlight
        if hasattr(trial_data, 'primary_outcome') and trial_data.primary_outcome:
            st.markdown("### 🎯 Primary Outcome")
            st.success(trial_data.primary_outcome)
        
        # Secondary outcomes in a nice list
        if hasattr(trial_data, 'secondary_outcomes') and trial_data.secondary_outcomes:
            st.markdown("### 📊 Secondary Outcomes")
            outcome_df = pd.DataFrame({
                'Outcome': trial_data.secondary_outcomes,
                'Index': range(1, len(trial_data.secondary_outcomes) + 1)
            })
            st.dataframe(outcome_df, hide_index=True)
        
        # Detailed description
        if hasattr(trial_data, 'detailed_description') and trial_data.detailed_description:
            st.markdown("### 📝 Detailed Description")
            st.write(trial_data.detailed_description)
        
        # Criteria sections
        TrialDisplayHandler._display_criteria_sections(trial_data)
        
        # Show missing information
        TrialDisplayHandler._display_missing_info(trial_data)

    # Helper methods
    @staticmethod
    def _display_metrics(trial_data: Any) -> None:
        """Display key metrics in columns"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("NCT ID", trial_data.nctid)
        
        with col2:
            if hasattr(trial_data, 'participants') and trial_data.participants:
                st.metric("Participants", trial_data.participants)
            else:
                st.metric("Participants", "Not specified")
        
        with col3:
            if hasattr(trial_data, 'phase') and trial_data.phase:
                st.metric("Phase", " | ".join(trial_data.phase))
            else:
                st.metric("Phase", "Not specified")

    @staticmethod
    def _display_criteria_sections(trial_data: Any) -> None:
        """Display inclusion/exclusion criteria in sections"""
        if hasattr(trial_data, 'inclusion_criteria') and trial_data.inclusion_criteria:
            st.markdown("### Inclusion Criteria")
            for criteria in trial_data.inclusion_criteria:
                st.write(f"• {criteria}")
        
        if hasattr(trial_data, 'exclusion_criteria') and trial_data.exclusion_criteria:
            st.markdown("### Exclusion Criteria") 
            for criteria in trial_data.exclusion_criteria:
                st.write(f"• {criteria}")

    @staticmethod
    def _display_criteria_expandable(trial_data: Any) -> None:
        """Display criteria in expandable sections"""
        if (hasattr(trial_data, 'inclusion_criteria') and trial_data.inclusion_criteria) or \
           (hasattr(trial_data, 'exclusion_criteria') and trial_data.exclusion_criteria):
            with st.expander("✅ Eligibility Criteria"):
                if hasattr(trial_data, 'inclusion_criteria') and trial_data.inclusion_criteria:
                    st.markdown("**Inclusion Criteria:**")
                    for criteria in trial_data.inclusion_criteria:
                        st.write(f"• {criteria}")
                
                if hasattr(trial_data, 'exclusion_criteria') and trial_data.exclusion_criteria:
                    st.markdown("**Exclusion Criteria:**")
                    for criteria in trial_data.exclusion_criteria:
                        st.write(f"• {criteria}")

    @staticmethod
    def _display_overview_tab(trial_data: Any) -> None:
        """Display overview tab content"""
        st.markdown("### Trial Overview")
        
        if hasattr(trial_data, 'brief_summary') and trial_data.brief_summary:
            st.markdown("**Summary:**")
            st.write(trial_data.brief_summary)
        
        if hasattr(trial_data, 'detailed_description') and trial_data.detailed_description:
            st.markdown("**Detailed Description:**")
            st.write(trial_data.detailed_description)

    @staticmethod
    def _display_outcomes_tab(trial_data: Any) -> None:
        """Display outcomes tab content"""
        st.markdown("### Study Outcomes")
        
        if hasattr(trial_data, 'primary_outcome') and trial_data.primary_outcome:
            st.markdown("**Primary Outcome:**")
            st.write(trial_data.primary_outcome)
        
        if hasattr(trial_data, 'secondary_outcomes') and trial_data.secondary_outcomes:
            st.markdown("**Secondary Outcomes:**")
            for i, outcome in enumerate(trial_data.secondary_outcomes, 1):
                st.write(f"{i}. {outcome}")

    @staticmethod
    def _display_eligibility_tab(trial_data: Any) -> None:
        """Display eligibility tab content"""
        st.markdown("### Eligibility Criteria")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if hasattr(trial_data, 'inclusion_criteria') and trial_data.inclusion_criteria:
                st.markdown("**Inclusion Criteria:**")
                for criteria in trial_data.inclusion_criteria:
                    st.write(f"✅ {criteria}")
            else:
                st.write("No inclusion criteria specified")
        
        with col2:
            if hasattr(trial_data, 'exclusion_criteria') and trial_data.exclusion_criteria:
                st.markdown("**Exclusion Criteria:**")
                for criteria in trial_data.exclusion_criteria:
                    st.write(f"❌ {criteria}")
            else:
                st.write("No exclusion criteria specified")

    @staticmethod
    def _display_raw_data_tab(trial_data: Any) -> None:
        """Display raw data tab content"""
        st.markdown("### Raw Data (JSON)")
        st.json(trial_data.model_dump(exclude_none=True))

    @staticmethod
    def _create_summary_dataframe(trial_data: Any) -> pd.DataFrame:
        """Create a summary dataframe from trial data"""
        summary_dict = {}
        for field, value in trial_data.model_dump().items():
            if value:  # Only include non-empty values
                if isinstance(value, list) and len(value) > 0:
                    summary_dict[field.title()] = ", ".join(value)
                elif not isinstance(value, list):
                    summary_dict[field.title()] = str(value)
        
        if summary_dict:
            return pd.DataFrame(list(summary_dict.items()), 
                              columns=['Field', 'Value'])
        return pd.DataFrame()

    @staticmethod
    def _display_missing_info(trial_data: Any) -> None:
        """Display information about missing fields"""
        missing_info = []
        fields_to_check = [
            ('detailed_description', 'Detailed Description'),
            ('inclusion_criteria', 'Inclusion Criteria'),
            ('exclusion_criteria', 'Exclusion Criteria'), 
            ('sponsors', 'Sponsors'),
            ('participants', 'Participant Count'),
            ('phase', 'Trial Phase')
        ]
        
        for field, display_name in fields_to_check:
            if not (hasattr(trial_data, field) and getattr(trial_data, field)):
                missing_info.append(display_name)
        
        if missing_info:
            st.markdown("### ⚠️ Missing Information")
            st.warning(f"The following information is not available: {', '.join(missing_info)}")

# Display function for other object types
class ComparisonDisplayHandler:
    """Handler for displaying trial comparison objects"""
    
    @staticmethod
    def display_comparison(comparison_data: Any) -> None:
        """Display trial comparison results"""
        st.subheader("🔄 Trial Comparison Results")
        
        # Header with trial IDs
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Trial 1:** {comparison_data.trial1_id}")
        with col2:
            st.info(f"**Trial 2:** {comparison_data.trial2_id}")
        
        # Similarities
        if hasattr(comparison_data, 'similarities') and comparison_data.similarities:
            st.markdown("### ✅ Key Similarities")
            for similarity in comparison_data.similarities:
                st.write(f"• {similarity}")
        
        # Differences
        if hasattr(comparison_data, 'differences') and comparison_data.differences:
            st.markdown("### 🔄 Key Differences")
            for difference in comparison_data.differences:
                st.write(f"• {difference}")
        
        # Detailed comparisons in expandable sections
        comparison_fields = [
            ('phase_comparison', '🧪 Phase Comparison'),
            ('intervention_comparison', '💊 Intervention Comparison'),
            ('population_comparison', '👥 Population Comparison'),
            ('outcome_comparison', '🎯 Outcome Comparison')
        ]
        
        for field, title in comparison_fields:
            if hasattr(comparison_data, field) and getattr(comparison_data, field):
                with st.expander(title):
                    st.write(getattr(comparison_data, field))
        
        # Recommendation
        if hasattr(comparison_data, 'recommendation') and comparison_data.recommendation:
            st.markdown("### 💡 Recommendation")
            st.success(comparison_data.recommendation)

class ComplianceDisplayHandler:
    """Handler for displaying compliance report objects"""
    
    @staticmethod
    def display_compliance(compliance_data: Any) -> None:
        """Display compliance report"""
        st.subheader("📋 Compliance Report")
        
        # Status indicator
        status_colors = {
            "Compliant": "🟢",
            "Minor Issues": "🟡",
            "Major Issues": "🟠", 
            "Non-Compliant": "🔴"
        }
        
        if hasattr(compliance_data, 'overall_status'):
            status_icon = status_colors.get(compliance_data.overall_status, "⚪")
            st.markdown(f"## {status_icon} Status: {compliance_data.overall_status}")
        
        # Score display
        if hasattr(compliance_data, 'compliance_score'):
            st.metric("Compliance Score", f"{compliance_data.compliance_score}/100")
        
        # Issues display
        if hasattr(compliance_data, 'issues') and compliance_data.issues:
            st.markdown("### ⚠️ Issues Found")
            for issue in compliance_data.issues:
                severity_colors = {
                    "Low": "🟢",
                    "Medium": "🟡", 
                    "High": "🟠",
                    "Critical": "🔴"
                }
                severity_icon = severity_colors.get(issue.severity, "⚪")
                
                with st.expander(f"{severity_icon} {issue.category} ({issue.severity})"):
                    st.write(f"**Description:** {issue.description}")
                    st.write(f"**Recommendation:** {issue.recommendation}")
        
        # Summary
        if hasattr(compliance_data, 'summary') and compliance_data.summary:
            st.markdown("### 📝 Summary")
            st.info(compliance_data.summary)
        
        # Next steps
        if hasattr(compliance_data, 'next_steps') and compliance_data.next_steps:
            st.markdown("### 🚀 Next Steps")
            for i, step in enumerate(compliance_data.next_steps, 1):
                st.write(f"{i}. {step}")

# Usage in your main Streamlit app
def display_structured_data(data: Any, display_type: str = "card", data_category: str = "trial") -> None:
    """
    Main function to display structured data based on type
    
    Args:
        data: The Pydantic object to display
        display_type: Style of display ("basic", "card", "tab", "dataframe", "comprehensive")
        data_category: Type of data ("trial", "comparison", "compliance")
    """
    
    if data_category == "trial":
        if display_type == "basic":
            TrialDisplayHandler.display_basic_format(data)
        elif display_type == "card":
            TrialDisplayHandler.display_card_format(data)
        elif display_type == "tab":
            TrialDisplayHandler.display_tab_format(data)
        elif display_type == "dataframe":
            TrialDisplayHandler.display_dataframe_format(data)
        elif display_type == "comprehensive":
            TrialDisplayHandler.display_comprehensive_format(data)
        else:
            TrialDisplayHandler.display_card_format(data)  # Default
            
    elif data_category == "comparison":
        ComparisonDisplayHandler.display_comparison(data)
        
    elif data_category == "compliance":
        ComplianceDisplayHandler.display_compliance(data)


class DevDisplayHandler:

    @staticmethod
    def basic_message():
        # Progress bar simulation
        st.subheader("🛠️ Feature In Progress")
        progress_bar = st.progress(0)
        for i in range(50):  
            progress_bar.progress(i + 1)
        st.write("Development Status: 50% Complete")

        with st.expander("🔍 What's Coming Next?"):
            st.write("""
            • NCT ID + Text based search \n
            • Enhanced Trials Summarization
            """)
