import requests
import json
import pandas as pd

def search_employees_one_row_per_employee_dedup(
    query,
    country_filter=None,
    location_filter=None,
    max_to_fetch=5
):
    """
    Search employees by:
      - 'query' (e.g. 'CEO', 'CEO OR CFO', etc.), 
      - optional 'country_filter' (e.g. 'South Africa'), 
      - optional 'location_filter' (e.g. 'Johannesburg, Gauteng, South Africa').

    In the final DataFrame (one row per employee):
      - Keep: ID, Name, Headline/Title, Location, Country, URL, Industry, experience_count, summary
      - Include: deduplicated Experiences (with 'duration'), Educations, and Skills
      - Remove: first_name, last_name
    """

    # 1) Build the Elasticsearch DSL query
    must_clauses = []

    # a) The nested query for experience titles
    must_clauses.append({
        "nested": {
            "path": "member_experience_collection",
            "query": {
                "query_string": {
                    "query": query,
                    "default_field": "member_experience_collection.title",
                    "default_operator": "and"
                }
            }
        }
    })

    # b) If user wants to filter by a specific country (exact match)
    if country_filter:
        must_clauses.append({
            "term": {
                "country": country_filter
            }
        })

    # c) If user wants to filter by a specific location (phrase match)
    if location_filter:
        must_clauses.append({
            "match_phrase": {
                "location": location_filter
            }
        })

    # Combine into a bool query
    payload = {
        "query": {
            "bool": {
                "must": must_clauses
            }
        }
    }

    # 2) Send the search request
    search_url = "https://api.coresignal.com/cdapi/v1/professional_network/employee/search/es_dsl"

    headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer eyJhbGciOiJFZERTQSIsImtpZCI6IjEwYTYwZWRhLWNhNzEtMTIxZS1jY2JhLTBmNjRjMzg4Yjg0ZCJ9.eyJhdWQiOiJheW9iYS5tZSIsImV4cCI6MTc3MzEwNjAyMSwiaWF0IjoxNzQxNTQ5MDY5LCJpc3MiOiJodHRwczovL29wcy5jb3Jlc2lnbmFsLmNvbTo4MzAwL3YxL2lkZW50aXR5L29pZGMiLCJuYW1lc3BhY2UiOiJyb290IiwicHJlZmVycmVkX3VzZXJuYW1lIjoiYXlvYmEubWUiLCJzdWIiOiI5Nzg4ZDg5Ni0yNzBjLTU4NjgtMTY0Mi05MWFiZDk0MGEwODYiLCJ1c2VyaW5mbyI6eyJzY29wZXMiOiJjZGFwaSJ9fQ.BeR_ci_7346iPkfP64QZCwxILa1v1_HGIE1SdhOl9qHtM_HcwiiWIf26DNhcDPl7Bs16JAEfjBntMoyJymtYDA'
    }

    resp = requests.post(search_url, headers=headers, json=payload)
    resp.raise_for_status()
    employee_ids = resp.json()

    if not isinstance(employee_ids, list):
        print("Unexpected structure in search response.")
        return pd.DataFrame()

    # 3) Collect data for each employee ID
    rows = []
    for emp_id in employee_ids[:max_to_fetch]:
        collect_url = f"https://api.coresignal.com/cdapi/v1/professional_network/employee/collect/{emp_id}"
        r = requests.get(collect_url, headers=headers)
        r.raise_for_status()

        employee = r.json()

        # Basic fields
        id_val = employee.get('id')
        name_val = employee.get('name')
        headline_val = employee.get('title')
        location_val = employee.get('location')
        country_val = employee.get('country')
        url_val = employee.get('url')
        industry_val = employee.get('industry')
        experience_count_val = employee.get('experience_count')
        summary_val = employee.get('summary')

        # ----- EXPERIENCE (deduplicate) -----
        raw_exps = employee.get('member_experience_collection', [])
        unique_exps = []
        seen_exps = set()
        for exp in raw_exps:
            key = (
                exp.get('title', 'N/A'),
                exp.get('company_name', 'N/A'),
                exp.get('date_from', 'N/A'),
                exp.get('date_to', 'N/A')
            )
            if key not in seen_exps:
                seen_exps.add(key)
                unique_exps.append(exp)

        experiences_str = "\n".join(
            f"Role: {exp.get('title','N/A')} | Company: {exp.get('company_name','N/A')} "
            f"| From: {exp.get('date_from','N/A')} | To: {exp.get('date_to','N/A')} "
            f"| Duration: {exp.get('duration','N/A')}"
            for exp in unique_exps
        )

        # ----- EDUCATION (deduplicate) -----
        raw_edu = employee.get('member_education_collection', [])
        unique_edu = []
        seen_edu = set()
        for edu in raw_edu:
            key = (
                edu.get('title', 'N/A'),
                edu.get('subtitle', 'N/A'),
                edu.get('date_from', 'N/A'),
                edu.get('date_to', 'N/A')
            )
            if key not in seen_edu:
                seen_edu.add(key)
                unique_edu.append(edu)

        educations_str = "\n".join(
            f"Institution: {edu.get('title','N/A')} | Degree: {edu.get('subtitle','N/A')} "
            f"| From: {edu.get('date_from','N/A')} | To: {edu.get('date_to','N/A')}"
            for edu in unique_edu
        )

        # ----- SKILLS (deduplicate) -----
        raw_skills = employee.get('member_skills_collection', [])
        seen_skills = set()
        for skill_entry in raw_skills:
            skill_name = skill_entry.get('member_skill_list', {}).get('skill', 'N/A')
            if skill_name not in seen_skills:
                seen_skills.add(skill_name)

        skills_str = ", ".join(seen_skills) if seen_skills else ""

        # Build final row
        row = {
            "ID": id_val,
            "Name": name_val,
            "Headline/Title": headline_val,
            "Location": location_val,
            "Country": country_val,
            "URL": url_val,
            "Industry": industry_val,
            "Experience Count": experience_count_val,
            "Summary": summary_val,
            "Experiences": experiences_str,
            "Educations": educations_str,
            "Skills": skills_str
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    user_query = "(Chief Financial Officer) OR (CFO)"
    country = "South Africa"
    location = "Johannesburg"
    df_employees = search_employees_one_row_per_employee_dedup(
        query=user_query,
        country_filter=country,  
        location_filter=location,
        max_to_fetch=15
    )
