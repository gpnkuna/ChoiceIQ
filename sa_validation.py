"""South African occupation validation metadata for ChoiceIQ.

Sources supplied by the project owner:
- Department of Higher Education and Training, National List of Occupations in High Demand: 2024.
- Department of Home Affairs, Critical Skills List, Government Gazette No. 49402, 3 October 2023.

This validation is intentionally separate from the recommendation score. It indicates whether an
ESCO career has an exact or closely related occupation in the supplied South African official lists.
"""

SA_VALIDATION = {
    "Ict Security Administrator": ("Related official occupation", "ICT Security Specialist", True, True),
    "Digital Forensics Expert": ("Related official occupation", "ICT Security Specialist", True, True),
    "Cyber Incident Responder": ("Related official occupation", "ICT Security Specialist", True, True),
    "Software Tester": ("Related official occupation", "Computer Quality Assurance Analyst", False, True),
    "Data Engineer": ("Not directly matched in supplied lists", "", False, False),
    "Data Scientist": ("Exact official occupation", "Data Scientist", True, True),
    "Mobile Application Developer": ("Related official occupation", "Applications Programmer", False, True),
    "Cloud Architect": ("Related official occupation", "Computer Network and Systems Engineer", True, True),
    "Cloud Engineer": ("Related official occupation", "Computer Network and Systems Engineer", True, True),
    "Artificial Intelligence Engineer": ("Not directly matched in supplied lists", "", False, False),
    "Ict Technician": ("Not directly matched in supplied lists", "", False, False),
    "Ethical Hacker": ("Related official occupation", "ICT Security Specialist", True, True),
    "Cybersecurity Risk Manager": ("Related official occupation", "ICT Security Specialist", True, True),
    "Ict Network Administrator": ("Related official occupation", "Systems Administrator / Network Analyst", True, True),
    "Database Administrator": ("Related official occupation", "Database Designer and Administrator", True, False),
    "User Interface Designer": ("Not directly matched in supplied lists", "", False, False),
    "Ict System Administrator": ("Related official occupation", "Systems Administrator", True, False),
    "Ict Security Technician": ("Related official occupation", "ICT Security Specialist", True, True),
    "Ict System Analyst": ("Exact official occupation", "ICT Systems Analyst", True, True),
    "Ict Help Desk Agent": ("Not directly matched in supplied lists", "", False, False),
    "Database Developer": ("Not directly matched in supplied lists", "", False, False),
    "Web Developer": ("Exact official occupation", "Web Developer", True, False),
    "Cloud Devops Engineer": ("Related official occupation", "Computer Network and Systems Engineer / Systems Administrator", True, True),
    "Ict Network Engineer": ("Related official occupation", "Computer Network and Systems Engineer", True, True),
    "Data Analyst": ("Not directly matched in supplied lists", "", False, False),
    "Ict Network Architect": ("Related official occupation", "Computer Network and Systems Engineer", True, True),
    "Ict System Architect": ("Related official occupation", "ICT Systems Analyst", True, True),
    "Computer Scientist": ("Not directly matched in supplied lists", "", False, False),
    "Software Developer": ("Exact official occupation", "Software Developer", True, True),
    "Ict Business Analyst": ("Not directly matched in supplied lists", "", False, False),
}
