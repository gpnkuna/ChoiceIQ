from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
RAW = BASE / "Data" / "raw"
PROC = BASE / "Data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

FEATURES = {
    "data_analytics": ["data", "analytics", "statistic", "visualis", "database", "information model", "business intelligence", "data mining"],
    "programming": ["program", "code", "coding", "software development", "scripting", "algorithm", "debug", "application development"],
    "mathematics": ["math", "statistic", "probability", "quantitative", "algorithm"],
    "cybersecurity": ["security", "cyber", "threat", "vulnerab", "forensic", "incident", "risk", "penetration", "ethical hack", "encryption"],
    "networking": ["network", "telecommunication", "router", "switch", "tcp", "infrastructure", "connectivity"],
    "problem_solving": ["problem", "troubleshoot", "diagnos", "analyse", "analysis", "resolve", "solution", "test", "quality"],
    "creativity_design": ["design", "user interface", "user experience", "creative", "prototype", "usability", "visual"],
    "systems_cloud": ["system", "cloud", "server", "operating system", "virtual", "devops", "deployment", "architecture", "configuration"],
    "communication_business": ["communicat", "client", "customer", "business", "requirement", "stakeholder", "consult", "support", "document"],
}

PREFERRED = [
    "data analyst", "data scientist", "data engineer", "software developer", "web developer", "mobile application developer",
    "artificial intelligence engineer", "computer scientist", "ICT business analyst", "ICT system analyst", "ICT system architect",
    "cloud engineer", "cloud architect", "cloud DevOps engineer", "database administrator", "database developer",
    "ICT network administrator", "ICT network architect", "ICT network engineer", "ICT system administrator",
    "ICT security administrator", "cyber incident responder", "cybersecurity risk manager", "digital forensics expert", "ethical hacker",
    "ICT security technician", "ICT help desk agent", "ICT technician", "software tester", "user interface designer",
]

# Transparent rule-based priors derived from occupation titles. They complement ESCO skills
# when a short title captures an important theme that is under-represented in skill text.
TITLE_PRIORS = {
    "data": {"data_analytics": 1.0, "mathematics": 0.45, "problem_solving": 0.45},
    "scientist": {"data_analytics": 0.8, "mathematics": 0.65, "programming": 0.45},
    "engineer": {"problem_solving": 0.65, "systems_cloud": 0.45},
    "software": {"programming": 1.0, "problem_solving": 0.55, "creativity_design": 0.3},
    "developer": {"programming": 1.0, "problem_solving": 0.45, "creativity_design": 0.35},
    "programmer": {"programming": 1.0, "problem_solving": 0.45},
    "web": {"programming": 0.85, "creativity_design": 0.7},
    "mobile": {"programming": 0.9, "creativity_design": 0.55},
    "artificial intelligence": {"programming": 0.85, "data_analytics": 0.75, "mathematics": 0.75},
    "business analyst": {"communication_business": 1.0, "problem_solving": 0.7, "data_analytics": 0.45},
    "system analyst": {"systems_cloud": 0.8, "problem_solving": 0.8, "communication_business": 0.55},
    "system architect": {"systems_cloud": 1.0, "problem_solving": 0.65, "creativity_design": 0.35},
    "cloud": {"systems_cloud": 1.0, "networking": 0.65, "programming": 0.35},
    "devops": {"systems_cloud": 1.0, "programming": 0.75, "networking": 0.55},
    "database": {"data_analytics": 0.85, "systems_cloud": 0.7, "programming": 0.45},
    "network": {"networking": 1.0, "systems_cloud": 0.8, "problem_solving": 0.55},
    "security": {"cybersecurity": 1.0, "problem_solving": 0.65, "systems_cloud": 0.45},
    "cyber": {"cybersecurity": 1.0, "problem_solving": 0.7},
    "forensic": {"cybersecurity": 1.0, "problem_solving": 0.85, "data_analytics": 0.45},
    "ethical hacker": {"cybersecurity": 1.0, "problem_solving": 0.8, "programming": 0.45},
    "help desk": {"communication_business": 0.9, "problem_solving": 0.8, "systems_cloud": 0.65},
    "technician": {"systems_cloud": 0.75, "problem_solving": 0.75, "communication_business": 0.4},
    "tester": {"problem_solving": 1.0, "programming": 0.65, "communication_business": 0.35},
    "user interface": {"creativity_design": 1.0, "communication_business": 0.55, "programming": 0.35},
}


def add_title_priors(label: str, vals: dict) -> dict:
    low = label.lower()
    for token, boosts in TITLE_PRIORS.items():
        if token in low:
            for feat, boost in boosts.items():
                vals[feat] += boost
    return vals


def main():
    occ = pd.read_csv(RAW / "occupations_en.csv", usecols=["conceptUri", "iscoGroup", "preferredLabel", "description", "code"])
    rel = pd.read_csv(RAW / "occupationSkillRelations_en.csv", usecols=["occupationUri", "occupationLabel", "relationType", "skillLabel"])
    occ["iscoGroup"] = occ["iscoGroup"].astype(str)

    ict = occ[occ["iscoGroup"].str.startswith(("25", "351", "3522"))].copy()
    ict = ict[ict.preferredLabel.isin(PREFERRED)].drop_duplicates("conceptUri")
    rels = rel[rel.occupationUri.isin(ict.conceptUri)].copy()

    rows = []
    for _, o in ict.iterrows():
        r = rels[rels.occupationUri.eq(o.conceptUri)]
        weighted = []
        for _, x in r.iterrows():
            label = str(x.skillLabel).lower()
            wt = 1.0 if str(x.relationType).lower() == "essential" else 0.55
            weighted.append((label, wt))

        desc = (str(o.description) if pd.notna(o.description) else "").lower()
        vals, matched = {}, {}
        for feat, kws in FEATURES.items():
            score, evidence = 0.0, []
            for label, wt in weighted:
                if any(k in label for k in kws):
                    score += wt
                    evidence.append(label)
            if any(k in desc for k in kws):
                score += 0.75
            vals[feat] = score
            matched[feat] = evidence[:5]

        vals = add_title_priors(str(o.preferredLabel), vals)
        rows.append({
            "career": o.preferredLabel.title(),
            "esco_label": o.preferredLabel,
            "isco_group": o.iscoGroup,
            "esco_code": o.code,
            "description": o.description,
            "essential_skill_count": int((r.relationType == "essential").sum()),
            "optional_skill_count": int((r.relationType != "essential").sum()),
            **vals,
        })

    df = pd.DataFrame(rows)
    feat_cols = list(FEATURES)
    for c in feat_cols:
        mx = df[c].max()
        if mx > 0:
            df[c] = (df[c] / mx).round(4)

    from sa_validation import SA_VALIDATION
    validation_rows = []
    for career in df["career"]:
        validation_rows.append(SA_VALIDATION.get(career, ("Not directly matched in supplied lists", "", False, False)))
    df[["sa_validation_status", "sa_official_match", "sa_high_demand_2024", "sa_critical_skills_2023"]] = pd.DataFrame(validation_rows, index=df.index)
    df["sa_any_validation"] = df["sa_high_demand_2024"] | df["sa_critical_skills_2023"]
    df["sa_validation_source"] = np.select(
        [df["sa_high_demand_2024"] & df["sa_critical_skills_2023"], df["sa_high_demand_2024"], df["sa_critical_skills_2023"]],
        ["DHET 2024 High Demand + DHA 2023 Critical Skills", "DHET 2024 High Demand", "DHA 2023 Critical Skills"],
        default="No direct match in supplied official lists",
    )

    df = df.sort_values("career").reset_index(drop=True)
    df.to_csv(PROC / "choiceiq_careers.csv", index=False)
    df[["career", "sa_validation_status", "sa_official_match", "sa_high_demand_2024", "sa_critical_skills_2023", "sa_validation_source"]].to_csv(PROC / "south_africa_validation.csv", index=False)

    rels2 = rels.merge(ict[["conceptUri", "preferredLabel"]], left_on="occupationUri", right_on="conceptUri", suffixes=("", "_occ"))
    rels2[["occupationLabel", "relationType", "skillLabel"]].to_csv(PROC / "choiceiq_skill_evidence.csv", index=False)
    print(f"Created {len(df)} careers with {len(feat_cols)} engineered features")


if __name__ == "__main__":
    main()
