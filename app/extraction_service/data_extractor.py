def extract_trial_info(trial):
    ps = trial.get("protocolSection", {})

    # Identifiers
    idm = ps.get("identificationModule", {})
    identifiers = {
        "nctId": idm.get("nctId"),
        "orgStudyId": idm.get("orgStudyIdInfo", {}).get("id"),
        "acronym": idm.get("acronym")
    }

    # Titles
    titles = {
        "briefTitle": idm.get("briefTitle"),
        "officialTitle": idm.get("officialTitle")
    }

    # Status
    sm = ps.get("statusModule", {})
    status = {
        "overallStatus": sm.get("overallStatus"),
        "startDate": sm.get("startDateStruct", {}).get("date"),
        "completionDate": sm.get("completionDateStruct", {}).get("date")
    }

    # Sponsor/Collaborators
    scm = ps.get("sponsorCollaboratorsModule", {})
    sponsor = {
        "leadSponsor": scm.get("leadSponsor", {}),
        "responsibleParty": scm.get("responsibleParty", {})
    }

    # Description
    description = {"briefSummary": ps.get("descriptionModule", {}).get("briefSummary")}

    # Design
    dm = ps.get("designModule", {})
    design = {
        "studyType": dm.get("studyType"),
        "observationalModel": dm.get("designInfo", {}).get("observationalModel"),
        "timePerspective": dm.get("designInfo", {}).get("timePerspective"),
        "enrollment": dm.get("enrollmentInfo", {}).get("count")
    }

    # Arms/Interventions
    aim = ps.get("armsInterventionsModule", {})
    arms = aim.get("armGroups") or aim.get("interventions", [])

    # Outcomes
    om = ps.get("outcomesModule", {})
    outcomes = {
        "primaryOutcomes": om.get("primaryOutcomes", []),
        "secondaryOutcomes": om.get("secondaryOutcomes", [])
    }

    # Eligibility
    em = ps.get("eligibilityModule", {})
    criteria = em.get("eligibilityCriteria", "")
    inclusion = []
    exclusion = []
    if "Inclusion Criteria:" in criteria:
        parts = criteria.split("Exclusion Criteria:")
        inclusion = parts[0].replace("Inclusion Criteria:", "").strip()
        if len(parts) > 1:
            exclusion = parts[1].strip()

    eligibility = {
        "inclusionCriteria": inclusion,
        "exclusionCriteria": exclusion,
        "sex": em.get("sex"),
        "age": em.get("minimumAge")
    }

    # Location
    clm = ps.get("contactsLocationsModule", {})
    locations = [
        {
            "facility": loc.get("facility"),
            "city": loc.get("city"),
            "country": loc.get("country")
        }
        for loc in clm.get("locations", [])
    ]

    return {
        "Identifiers": identifiers,
        "Titles": titles,
        "Status": status,
        "Sponsor/Collaborators": sponsor,
        "Description": description,
        "Design": design,
        "Arms/Interventions": arms,
        "Outcomes": outcomes,
        "Eligibility": eligibility,
        "Locations": locations
    }


def build_trial_text(trial):
    parts = []
    # NCT ID

    if trial["Identifiers"].get("nctId"):
        parts.append("NCT ID: " + trial["Identifiers"]["nctId"])
    
    # Titles
    if trial["Titles"].get("briefTitle"):
        parts.append("Brief Title: " + trial["Titles"]["briefTitle"])
    if trial["Titles"].get("officialTitle"):
        parts.append("Official Title: " + trial["Titles"]["officialTitle"])

    # Status
    status = trial.get("Status", {})
    if status:
        parts.append(f"Status: {status.get('overallStatus','')}, "
                     f"Start: {status.get('startDate','')}, "
                     f"Completion: {status.get('completionDate','')}")

    # Description
    if trial["Description"].get("briefSummary"):
        parts.append("Summary: " + trial["Description"]["briefSummary"])
        # parts.append("Detailed Description: "+ trial["Description"].get("detailedDescription", "Not Available"))

    # Design
    design = trial.get("Design", {})
    if design:
        parts.append(f"Design: {design.get('studyType','')} "
                     f"({design.get('observationalModel','')}, "
                     f"{design.get('timePerspective','')}), "
                     f"Enrollment: {design.get('enrollment','')}")

    # Arms
    for arm in trial.get("Arms/Interventions", []):
        parts.append(f"Intervention: {arm.get('name','')} - {arm.get('description','')}")

    # Outcomes
    for out in trial["Outcomes"].get("primaryOutcomes", []):
        parts.append(f"Primary Outcome: {out.get('measure','')} - {out.get('description','')} "
                     f"({out.get('timeFrame','')})")

    # Eligibility
    elig = trial.get("Eligibility", {})
    if elig.get("inclusionCriteria"):
        parts.append("Inclusion: " + elig["inclusionCriteria"])
    if elig.get("exclusionCriteria"):
        parts.append("Exclusion: " + elig["exclusionCriteria"])

    # Locations
    for loc in trial.get("Locations", []):
        parts.append(f"Location: {loc.get('facility','')} - {loc.get('city','')}, {loc.get('country','')}")

    return "\n".join(parts)


