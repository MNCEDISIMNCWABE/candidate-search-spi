import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import requests
import logging
import warnings
import re
from typing import List, Optional
import io
import hashlib
import pickle
import os
from datetime import datetime, timedelta

# Set up logging and ignore warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User authentication functions
def make_hashed_password(password):
    """Create a hashed version of the password."""
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_password(stored_password, input_password):
    """Check if the input password matches the stored password."""
    return stored_password == make_hashed_password(input_password)

def save_users(users_dict):
    """Save the users dictionary to a file."""
    with open('users.pkl', 'wb') as f:
        pickle.dump(users_dict, f)

def load_users():
    """Load the users dictionary from a file."""
    if os.path.exists('users.pkl'):
        with open('users.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        # Create a default admin user
        users = {
            'admin': {
                'password': make_hashed_password('SPI123@_'),
                'email': 'admin@example.com',
                'created_at': datetime.now(),
                'role': 'admin'
            }
        }
        save_users(users)
        return users

# API functions
import pandas as pd
import numpy as np
import requests
import logging
import warnings
import re
import os


# Set up logging and ignore warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_employees_one_row_per_employee_dedup(
    query,
    country_filter=None,
    location_filter=None,
    company_filter=None,
    university_filter=None,
    industry_filter=None,
    skills_filter=None,
    certifications_filter=None,
    languages_filter=None,
    max_to_fetch=None
):
    """
    Search employees by:
      - 'query' (e.g. 'CEO', 'CEO OR CFO', etc.)
      - Optional filters:
            country_filter (e.g. 'South Africa'),
            location_filter (e.g. 'Johannesburg, Gauteng, South Africa'),
            company_filter (search in company names),
            university_filter (search in university names),
            industry_filter (search in the top-level industry field),
            skills_filter (search in skills),
            certifications_filter (search in certifications),
            languages_filter (search in languages),
            projects_filter (provided for consistency but not used in the search query).

    In the final DataFrame (one row per employee):
      - Keeps: ID, Name, Headline/Title, Location, Country, URL, Canonical_URL, Industry,
               Experience Count, Summary.
      - Includes: deduplicated Experiences (with duration), Educations, Skills, Certifications,
                  Languages, and Projects.
    """
    # Build the list of must clauses.
    must_clauses = []

    # Base clause: search in experience title
    must_clauses.append({
        "nested": {
            "path": "experience",
            "query": {
                "query_string": {
                    "query": query,
                    "default_field": "experience.position_title",
                    "default_operator": "and"
                }
            }
        }
    })

    # Additional filter: Company Name (in experience)
    if company_filter:
        must_clauses.append({
            "nested": {
                "path": "experience",
                "query": {
                    "query_string": {
                        "query": company_filter,
                        "default_field": "experience.company_name",
                        "default_operator": "or"
                    }
                }
            }
        })

    # Additional filter: University Name (in education)
    if university_filter:
        must_clauses.append({
            "nested": {
                "path": "education",
                "query": {
                    "query_string": {
                        "query": university_filter,
                        "default_field": "education.institution_name",
                        "default_operator": "or"
                    }
                }
            }
        })

    # Additional filter: Industry (in experience)
    if industry_filter:
        must_clauses.append({
            "nested": {
                "path": "experience",
                "query": {
                    "query_string": {
                        "query": industry_filter,
                        "default_field": "experience.company_industry",
                        "default_operator": "or"
                    }
                }
            }
        })

    # Additional filter: Skills (in inferred_skills)
    if skills_filter:
        must_clauses.append({
            "query_string": {
                "query": skills_filter,
                "default_field": "inferred_skills",
                "default_operator": "or"
            }
        })

    # Additional filter: Certifications
    if certifications_filter:
        must_clauses.append({
            "nested": {
                "path": "certifications",
                "query": {
                    "query_string": {
                        "query": certifications_filter,
                        "default_field": "certifications.title",
                        "default_operator": "or"
                    }
                }
            }
        })

    # Additional filter: Languages
    if languages_filter:
        must_clauses.append({
            "nested": {
                "path": "languages",
                "query": {
                    "query_string": {
                        "query": languages_filter.lower(),
                        "default_field": "languages.language",
                        "default_operator": "or"
                    }
                }
            }
        })

    # Exclude patterns in titles
    exclude_patterns = ["PA to", "Assistant to", "Personal Assistant", "EA to", "Executive Assistant to","Head of the Office of the CFO","Head of the Office of the CEO"]
    must_not_clauses = [
        {
            "nested": {
                "path": "experience",
                "query": {
                    "query_string": {
                        "query": f"experience.position_title:({pattern})",
                        "default_operator": "or"
                    }
                }
            }
        }
        for pattern in exclude_patterns
    ]

    # Build the complete payload with country and location filters added.
    payload = {
        "query": {
            "bool": {
                "must": must_clauses,
                "must_not": must_not_clauses
            }
        }
    }

    if country_filter:
        payload["query"]["bool"]["must"].append({
            "query_string": {
                "query": country_filter,
                "default_field": "location_country",
                "default_operator": "and"
            }
        })

    if location_filter:
        payload["query"]["bool"]["must"].append({
            "query_string": {
                "query": location_filter,
                "default_field": "location_full",
                "default_operator": "and"
            }
        })

    # Uncomment for debugging:
    # print(json.dumps(payload, indent=2))

    # Send the search request.
    search_url = "https://api.coresignal.com/cdapi/v1/multi_source/employee/search/es_dsl"
    headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer eyJhbGciOiJFZERTQSIsImtpZCI6IjMzNjEyYzA1LWQ2MDYtYzllYy0zNGVjLWRiYmJiNGI0ZjgyMCJ9.eyJhdWQiOiJtdWx0aWNob2ljZS5jby56YSIsImV4cCI6MTc3MzQwNjg1OCwiaWF0IjoxNzQxODQ5OTA2LCJpc3MiOiJodHRwczovL29wcy5jb3Jlc2lnbmFsLmNvbTo4MzAwL3YxL2lkZW50aXR5L29pZGMiLCJuYW1lc3BhY2UiOiJyb290IiwicHJlZmVycmVkX3VzZXJuYW1lIjoibXVsdGljaG9pY2UuY28uemEiLCJzdWIiOiI5Nzg4ZDg5Ni0yNzBjLTU4NjgtMTY0Mi05MWFiZDk0MGEwODYiLCJ1c2VyaW5mbyI6eyJzY29wZXMiOiJjZGFwaSJ9fQ.GFaoIY_j8e3TKs9-iQ0H6O7NVz87T3Z7ZWIWPRHo17IrWqmehNvvJ8sD3BMaDVatHs9rr9C3hpUykkwS53HrAw' 
    }

    resp = requests.post(search_url, headers=headers, json=payload)
    resp.raise_for_status()
    employee_ids = resp.json()

    if not isinstance(employee_ids, list):
        print("Unexpected structure in search response.")
        return pd.DataFrame()

    # Collect data for each employee ID.
    rows = []
    for emp_id in employee_ids[:max_to_fetch]:
        
        collect_url = f"https://api.coresignal.com/cdapi/v1/multi_source/employee/collect/{emp_id}"
        r = requests.get(collect_url, headers=headers)
        r.raise_for_status()
        employee = r.json()

        # Basic fields
        id_val = employee.get("id")
        name_val = employee.get("full_name")
        headline_val = employee.get("headline")
        location_val = employee.get("location_full")
        country_val = employee.get("location_country")
        url_val = employee.get("linkedin_url")
        canonical_url = employee.get("linkedin_url")  # Using LinkedIn URL as canonical
        industry_val = None  # Not available in top level, will need to be extracted from experience
        experience_count_val = len(employee.get("experience", []))
        summary_val = employee.get("summary")
        
        # Get email information
        primary_email = employee.get("primary_professional_email")
        
        # Get all email addresses from collection
        email_collection = employee.get("professional_emails_collection", [])
        all_emails = [email_info.get("professional_email") for email_info in email_collection if email_info.get("professional_email")]
        all_emails_str = ", ".join(all_emails) if all_emails else ""

        # ----- EXPERIENCE (deduplicate) -----
        raw_exps = employee.get("experience", [])
        unique_exps = []
        seen_exps = set()
        company_industries = set()  # Set to collect unique industries
        for exp in raw_exps:
            key = (
                exp.get("position_title", "N/A"),
                exp.get("company_name", "N/A"),
                exp.get("date_from", "N/A"),
                exp.get("date_to", "N/A")
            )
            if key not in seen_exps:
                seen_exps.add(key)
                unique_exps.append(exp)
                # Add industry to the set if it exists
                if exp.get("company_industry"):
                    company_industries.add(exp.get("company_industry"))

        experiences_str = "\n".join(
            f"Role: {exp.get('position_title','N/A')} | Company: {exp.get('company_name','N/A')} | From: {exp.get('date_from','N/A')} | To: {exp.get('date_to','N/A')} | Duration: {exp.get('duration_months','N/A')} months"
            for exp in unique_exps
        )

        # Create a formatted string of industries
        company_industry_str = " | ".join(sorted(company_industries)) if company_industries else "N/A"

        # ----- EDUCATION (deduplicate) -----
        raw_edu = employee.get("education", [])
        unique_edu = []
        seen_edu = set()
        for edu in raw_edu:
            key = (
                edu.get("institution_name", "N/A"),
                edu.get("degree", "N/A"),
                str(edu.get("date_from_year", "N/A")),
                str(edu.get("date_to_year", "N/A"))
            )
            if key not in seen_edu:
                seen_edu.add(key)
                unique_edu.append(edu)
        educations_str = "\n".join(
            f"Institution: {edu.get('institution_name','N/A')} | Degree: {edu.get('degree','N/A')} | From: {edu.get('date_from_year','N/A')} | To: {edu.get('date_to_year','N/A')}"
            for edu in unique_edu
        )

        # ----- SKILLS (deduplicate) -----
        skills = employee.get("inferred_skills", [])
        skills_str = ", ".join(skills) if skills else ""

        # ----- CERTIFICATIONS (deduplicate) -----
        raw_certifications = employee.get("certifications", [])
        seen_certs = set()
        for cert in raw_certifications:
            cert_name = cert.get("title", "N/A")
            seen_certs.add(cert_name)
        certifications_str = ", ".join(seen_certs) if seen_certs else ""

        # ----- LANGUAGES (deduplicate) -----
        raw_languages = employee.get("languages", [])
        seen_langs = set()
        for lang in raw_languages:
            language_name = lang.get("language", "N/A")
            seen_langs.add(language_name)
        languages_str = ", ".join(seen_langs) if seen_langs else ""

        # ----- PROJECTS (deduplicate) -----
        raw_projects = employee.get("projects", [])
        seen_projects = set()
        for proj in raw_projects:
            proj_name = proj.get("name", "N/A")
            seen_projects.add(proj_name)
        projects_str = ", ".join([str(x) for x in seen_projects if x is not None]) if seen_projects else ""

        # ----- AWARDS (deduplicate) -----
        raw_awards = employee.get("awards", [])
        seen_awards = set()
        for award in raw_awards:
            award_name = award.get("title", "N/A")
            seen_awards.add(award_name)
        awards_str = ", ".join(seen_awards) if seen_awards else ""

        # ----- PATENTS (deduplicate) -----
        raw_patents = employee.get("patents", [])
        seen_patents = set()
        for patent in raw_patents:
            patent_name = patent.get("title", "N/A")
            seen_patents.add(patent_name)
        patents_str = ", ".join(seen_patents) if seen_patents else ""

        # ----- PUBLICATIONS (deduplicate) -----
        raw_publications = employee.get("publications", [])
        seen_publications = set()
        for pub in raw_publications:
            pub_name = pub.get("title", "N/A")
            seen_publications.add(pub_name)
        publications_str = ", ".join(seen_publications) if seen_publications else ""

        # ----- SALARY INFORMATION -----
        projected_base_salary_median = employee.get("projected_base_salary_median")
        projected_base_salary_currency = employee.get("projected_base_salary_currency")
        projected_base_salary_period = employee.get("projected_base_salary_period")
        
        salary_str = ""
        if projected_base_salary_median:
            salary_str = f"{projected_base_salary_currency}{projected_base_salary_median:,.2f} {projected_base_salary_period}"

        # Build the final row dictionary.
        row = {
            "ID": id_val,
            "Name": name_val,
            "Headline/Title": headline_val,
            "Location": location_val,
            "Country": country_val,
            "URL": url_val,
            "Primary Email": primary_email,
            "All Emails": all_emails_str,
            "Industry": company_industry_str, 
            "Experience Count": experience_count_val,
            "Summary": summary_val,
            "Experiences": experiences_str,
            "Educations": educations_str,
            "Skills": skills_str,
            "Certifications": certifications_str,
            "Languages": languages_str,
        }
        rows.append(row)

    # After the search API call
    df = pd.DataFrame(rows)

    return df

# Ranking functions
def build_user_text(row, text_columns: List[str]) -> str:
    """
    Combine relevant text fields into a single string for semantic comparison.
    """
    parts = []
    for col in text_columns:
        val = row.get(col)
        if pd.notnull(val):
            if isinstance(val, list):
                parts.append(' '.join(map(str, val)))
            else:
                parts.append(str(val))
    return " ".join(parts).strip()

def preprocess_text(text: str) -> str:
    """
    Clean and normalize text input.
    """
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  
                               u"\U0001F300-\U0001F5FF"  
                               u"\U0001F680-\U0001F6FF"  
                               u"\U0001F1E0-\U0001F1FF"  
                               u"\U00002500-\U00002BEF"  
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    text = ' '.join(text.strip().split())
    return text

def rank_candidates_semantic(
    df_employees: pd.DataFrame,
    job_description: str,
    text_columns: Optional[List[str]] = None,
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Rank candidates based on semantic similarity to a job description.
    Note: The default text columns include Certifications but intentionally omit Languages.
    """
    try:
        logger.info("Starting candidate ranking process...")
        df = df_employees.copy()
        
        if text_columns is None:
            text_columns = [
                'Summary', 'Experiences', 'Educations', 'Headline/Title',
                'Industry', 'Skills', 'Certifications'
            ]
            logger.debug(f"Using default text columns: {text_columns}")
        else:
            logger.debug(f"Using custom text columns: {text_columns}")

        logger.info("Combining candidate text fields...")
        df['combined_text'] = df.apply(lambda x: build_user_text(x, text_columns), axis=1)
        logger.info(f"Processed {len(df)} candidate profiles")

        logger.info("Filtering empty candidate texts...")
        initial_count = len(df)
        df['combined_text'] = df['combined_text'].replace(r'^\s*$', np.nan, regex=True)
        df = df.dropna(subset=['combined_text']).reset_index(drop=True)
        filtered_count = len(df)
        logger.info(f"Removed {initial_count - filtered_count} empty profiles, {filtered_count} remaining")

        if df.empty:
            logger.warning("No valid candidate texts found after preprocessing")
            return pd.DataFrame()

        logger.info(f"Initializing sentence transformer model: {model_name}")
        model = SentenceTransformer(model_name)
        
        logger.info("Preprocessing job description...")
        clean_jd = preprocess_text(job_description)
        logger.debug(f"Job description length: {len(clean_jd.split())} words")
        
        logger.info("Encoding job description...")
        job_embedding = model.encode(clean_jd, convert_to_tensor=True)
        logger.debug(f"Job embedding shape: {job_embedding.shape}")

        logger.info("Preprocessing candidate texts...")
        user_texts = df['combined_text'].apply(preprocess_text).tolist()
        logger.debug(f"First candidate text preview: {user_texts[0][:200]}...")
        
        logger.info(f"Encoding candidate texts in batches of {batch_size}...")
        user_embeddings = model.encode(
            user_texts,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=True
        )
        logger.info(f"Successfully encoded {len(user_texts)} candidate texts")
        logger.debug(f"Embeddings matrix shape: {user_embeddings.shape}")

        logger.info("Calculating cosine similarities...")
        similarities = util.cos_sim(job_embedding, user_embeddings)
        df['similarity_score'] = similarities.cpu().numpy().flatten()
        df['match_percentage'] = (df['similarity_score'] * 100).round(2).astype(str) + '%'
        
        min_score = df['similarity_score'].min()
        max_score = df['similarity_score'].max()
        logger.info(f"Similarity scores range: {min_score:.3f} - {max_score:.3f}")
        logger.debug(f"Score distribution:\n{df['similarity_score'].describe()}")

        logger.info("Sorting candidates by similarity score...")
        df_sorted = df.sort_values(by='similarity_score', ascending=False).reset_index(drop=True)
        df_sorted = df_sorted.drop('combined_text',axis=1)

        logger.info(f"Top candidate score: {df_sorted.iloc[0]['similarity_score']:.3f}")
        logger.info("Ranking process completed successfully")
        return df_sorted

    except Exception as e:
        logger.error(f"Error in ranking candidates: {str(e)}")
        raise

# Cache the model to avoid reloading
@st.cache_resource
def load_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

# Function to convert dataframe to Excel for download
def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Ranked Candidates', index=False)
    writer.close()
    processed_data = output.getvalue()
    return processed_data

# Login management functions
def login_page():
    st.title("SPI Executive Search")
    login_tab, signup_tab, forgot_tab = st.tabs(["Login", "Sign Up", "Forgot Password"])
    
    with login_tab:
        username_or_email = st.text_input("Username or Email", key="login_username_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            users = load_users()
            # Check if the input matches any username directly
            if username_or_email in users and check_password(users[username_or_email]['password'], password):
                st.session_state.logged_in = True
                st.session_state.username = username_or_email
                st.session_state.user_role = users[username_or_email].get('role', 'user')
                st.success(f"Welcome back, {username_or_email}!")
                st.rerun()
            else:
                # If not found by username, search by email
                found = False
                for username, user_data in users.items():
                    if user_data.get('email') == username_or_email and check_password(user_data['password'], password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_role = user_data.get('role', 'user')
                        st.success(f"Welcome back, {username}!")
                        found = True
                        st.rerun()
                        break
                
                if not found:
                    st.error("Invalid username/email or password")
    
    with signup_tab:
        if st.session_state.get('user_role') == 'admin' or not os.path.exists('users.pkl'):
            new_username = st.text_input("New Username", key="new_username")
            new_password = st.text_input("New Password", type="password", key="new_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            email = st.text_input("Email", key="email")
            
            if st.button("Sign Up"):
                users = load_users()
                # Check if username already exists
                if new_username in users:
                    st.error("Username already exists")
                # Check if email already exists
                elif any(user_data.get('email') == email for user_data in users.values()):
                    st.error("Email already in use")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                elif not new_username or not new_password:
                    st.error("Username and password cannot be empty")
                elif not email:
                    st.error("Email cannot be empty")
                else:
                    users[new_username] = {
                        'password': make_hashed_password(new_password),
                        'email': email,
                        'created_at': datetime.now(),
                        'role': 'user'
                    }
                    save_users(users)
                    st.success("Account created successfully! You can now login.")
        else:
            st.info("User registration is only managed by administrators. Please contact your administrator for access.")
    with forgot_tab:
        st.subheader("Reset Password")
        username_or_email = st.text_input("Enter your username or email", key="reset_username_email")
        new_password = st.text_input("New Password", type="password", key="reset_new_password")
        confirm_password = st.text_input("Confirm New Password", type="password", key="reset_confirm_password")
        
        if st.button("Reset Password", key="reset_password_button"):
            users = load_users()
            user_found = False
            username = None
            
            # Check if the input is a username
            if username_or_email in users:
                username = username_or_email
                user_found = True
            else:
                # Check if the input is an email
                for u, data in users.items():
                    if data.get('email', '').lower() == username_or_email.lower():
                        username = u
                        user_found = True
                        break
            
            if not user_found:
                st.error("No account found with the provided username or email.")
            elif new_password != confirm_password:
                st.error("New passwords do not match. Please try again.")
            else:
                # Update the password
                users[username]['password'] = make_hashed_password(new_password)
                save_users(users)
                st.success("Password has been reset successfully. You can now login with your new password.")    

def logout():
    if st.sidebar.button("Logout"):
        for key in ['logged_in', 'username', 'user_role']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

def admin_dashboard():
    st.title("Admin Dashboard - User Management")
    users = load_users()
    user_df = pd.DataFrame([
        {
            'Username': username,
            'Email': data['email'],
            'Created At': data['created_at'].strftime('%Y-%m-%d %H:%M:%S'),
            'Role': data.get('role', 'user')
        }
        for username, data in users.items()
    ])
    st.dataframe(user_df)
    st.subheader("Add New User")
    col1, col2 = st.columns(2)
    with col1:
        new_username = st.text_input("Username", key="admin_new_username")
        new_password = st.text_input("Password", type="password", key="admin_new_password")
    with col2:
        email = st.text_input("Email", key="admin_email")
        role = st.selectbox("Role", ["user", "admin"], key="admin_role")
    
    if st.button("Add User"):
        if new_username in users:
            st.error("Username already exists")
        elif not new_username or not new_password:
            st.error("Username and password cannot be empty")
        else:
            users[new_username] = {
                'password': make_hashed_password(new_password),
                'email': email,
                'created_at': datetime.now(),
                'role': role
            }
            save_users(users)
            st.success(f"User '{new_username}' added successfully")
            st.rerun()
    
    st.subheader("Delete User")
    username_to_delete = st.selectbox("Select User to Delete", list(users.keys()))
    if st.button("Delete User") and username_to_delete:
        if username_to_delete == st.session_state.username:
            st.error("You cannot delete your own account while logged in!")
        else:
            del users[username_to_delete]
            save_users(users)
            st.success(f"User '{username_to_delete}' deleted successfully")
            st.rerun()

def main():
    st.set_page_config(page_title="Candidate Search & Match", layout="wide")
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'ranked_results' not in st.session_state:
        st.session_state.ranked_results = None
    if not os.path.exists('users.pkl'):
        load_users()
    
    if st.session_state.logged_in:
        st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
        st.sidebar.write(f"Role: **{st.session_state.user_role}**")
        logout()
        if st.session_state.user_role == 'admin':
            pages = ["Candidate Search", "Admin Dashboard"]
            selected_page = st.sidebar.selectbox("Navigation", pages)
            if selected_page == "Admin Dashboard":
                admin_dashboard()
                return

    if not st.session_state.logged_in:
        login_page()
        return

    st.title("Candidate Search & Match")
    st.markdown("Find and rank the best candidates for a job position")
    tab1, tab2 = st.tabs(["Search Candidates", "Ranked Results"])
    
    with tab1:
        st.header("Search for Candidates")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Search Criteria")
            search_query = st.text_input("Job Title/Position", placeholder="e.g. '(Chief Financial Officer) OR (CFO)'")
            loc_col1, loc_col2 = st.columns(2)
            with loc_col1:
                country = st.text_input("Country", placeholder="e.g. South Africa")
            with loc_col2:
                location = st.text_input("City", placeholder="e.g. Johannesburg")
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                company_filter = st.text_input("Company Name", placeholder="e.g. PWC")
            with comp_col2:
                university_filter = st.text_input("University Name", placeholder="e.g. University of Cape Town")
            industry_col1, industry_col2 = st.columns(2)
            with industry_col1:
                industry_filter = st.text_input("Industry", placeholder="e.g. Accounting")
            with industry_col2:
                skills_filter = st.text_input("Skills", placeholder="e.g. business strategy, financial modeling")
            cert_col1, cert_col2 = st.columns(2)
            with cert_col1:
                certifications_filter = st.text_input("Certifications", placeholder="e.g. Assessor")
            with cert_col2:
                languages_filter = st.text_input("Languages", placeholder="e.g. English")
            slider_col, btn_col = st.columns([2, 1])
            with slider_col:
                max_results = st.slider("Maximum number of results", 1, 150, 15)
            with btn_col:
                st.write("")
                st.write("")
                search_button = st.button("Search Candidates")
            
            if search_button and search_query:
                with st.spinner("Searching for candidates..."):
                    st.session_state.ranked_results = None
                    results = search_employees_one_row_per_employee_dedup(
                        query=search_query,
                        country_filter=country if country else None,
                        location_filter=location if location else None,
                        company_filter=company_filter if company_filter else None,
                        university_filter=university_filter if university_filter else None,
                        industry_filter=industry_filter if industry_filter else None,
                        skills_filter=skills_filter if skills_filter else None,
                        certifications_filter=certifications_filter if certifications_filter else None,
                        languages_filter=languages_filter if languages_filter else None,
                        max_to_fetch=max_results
                    )
                    if results.empty:
                        st.error("No candidates found matching your criteria.")
                    else:
                        st.session_state.search_results = results
                        st.success(f"Found {len(results)} candidates!")
        
        with col2:
            st.subheader("Job Description")
            st.markdown("Provide a detailed job description to rank candidates against:")
            job_description = st.text_area(
                "Enter job description", 
                height=250,
                placeholder="Paste detailed job description here to rank candidates by relevance..."
            )
            rank_button = st.button("Rank Candidates")
            
            if rank_button:
                if st.session_state.search_results is None or st.session_state.search_results.empty:
                    st.error("Please search for candidates first before ranking.")
                elif not job_description:
                    st.warning("Please provide a job description for ranking candidates.")
                else:
                    with st.spinner("Ranking candidates..."):
                        load_model()
                        ranked_df = rank_candidates_semantic(
                            df_employees=st.session_state.search_results,
                            job_description=job_description,
                            model_name='all-MiniLM-L6-v2'
                        )
                        if ranked_df.empty:
                            st.error("Error occurred during ranking. Please try again.")
                        else:
                            st.session_state.ranked_results = ranked_df
                            st.success("Candidates ranked successfully! View results in the 'Ranked Results' tab.")
        
        if st.session_state.search_results is not None and not st.session_state.search_results.empty:
            st.subheader("Search Results")
            for i, row in st.session_state.search_results.iterrows():
                with st.expander(f"{row['Name']} - {row['Headline/Title']}"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"**Location:** {row['Location']}")
                        st.markdown(f"**Country:** {row['Country']}")
                        st.markdown(f"**Industry:** {row['Industry']}")
                        st.markdown(f"**Profile URL:** [Link]({row['URL']})")
                    with col2:
                        if pd.notnull(row['Summary']) and row['Summary']:
                            st.markdown("**Summary:**")
                            st.markdown(row['Summary'])
                        if pd.notnull(row['Skills']) and row['Skills']:
                            st.markdown("**Skills:**")
                            st.markdown(row['Skills'])
                    if pd.notnull(row['Experiences']) and row['Experiences']:
                        st.markdown("---")
                        st.markdown("### Experience Details")
                        experiences = row['Experiences'].split('\n')
                        for exp in experiences:
                            st.markdown(f"- {exp}")
                    if pd.notnull(row['Educations']) and row['Educations']:
                        st.markdown("---")
                        st.markdown("### Education Details")
                        educations = row['Educations'].split('\n')
                        for edu in educations:
                            st.markdown(f"- {edu}")

    with tab2:
        st.header("Ranked Candidates")
        if st.session_state.ranked_results is not None and not st.session_state.ranked_results.empty:
            export_columns = [
                'ID', 'Name', 'Headline/Title', 'Location', 'Country', 'URL', 
                'Primary Email', 'All Emails', 'Industry', 'Experience Count', 'Summary', 'Experiences', 
                'Educations', 'Skills', 'Certifications', 'Languages', 'similarity_score'
            ]
            export_df = st.session_state.ranked_results[
                [col for col in export_columns if col in st.session_state.ranked_results.columns]
            ].copy()
            if 'similarity_score' in export_df.columns:
                export_df['similarity_score'] = round(export_df['similarity_score'] * 100, 2)
            excel_data = to_excel(export_df)
            st.download_button(
                label="📥 Download Ranked Candidates (Excel)",
                data=excel_data,
                file_name='ranked_candidates.xlsx',
                mime='application/vnd.ms-excel',
            )
            
            st.subheader("Match Results")
            top_candidates = st.session_state.ranked_results.head(10)
            chart_data = pd.DataFrame({
                'Candidate': top_candidates['Name'],
                'Match Percentage': top_candidates['similarity_score'] * 100
            })
            st.bar_chart(chart_data.set_index('Candidate'))
            
            for i, row in st.session_state.ranked_results.iterrows():
                with st.expander(f"{row['Name']} - {row['Headline/Title']} (Match: {row['match_percentage']})"):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"**Match Score:** {row['match_percentage']}")
                        st.markdown(f"**Location:** {row['Location']}")
                        st.markdown(f"**Country:** {row['Country']}")
                        st.markdown(f"**Industry:** {row['Industry']}")
                        st.markdown(f"**Profile URL:** [Link]({row['URL']})")
                    with col2:
                        if pd.notnull(row['Summary']) and row['Summary']:
                            st.markdown("**Summary:**")
                            st.markdown(row['Summary'])
                        if pd.notnull(row['Skills']) and row['Skills']:
                            st.markdown("**Skills:**")
                            st.markdown(row['Skills'])
                    if pd.notnull(row['Experiences']) and row['Experiences']:
                        st.markdown("---")
                        st.markdown("### Experience Details")
                        experiences = row['Experiences'].split('\n')
                        for exp in experiences:
                            st.markdown(f"- {exp}")
                    if pd.notnull(row['Educations']) and row['Educations']:
                        st.markdown("---")
                        st.markdown("### Education Details")
                        educations = row['Educations'].split('\n')
                        for edu in educations:
                            st.markdown(f"- {edu}")
        else:
            st.info("No ranked results available. Please search for candidates and rank them first.")

    st.markdown("---")
    st.markdown("""
    **How to use this application:**
    1. Enter a job title and optionally other parameters in the search boxes (use OR for multiple terms)
    2. Use advanced filters to narrow down candidates
    3. Click "Search Candidates" to find matching profiles
    4. Enter a detailed job description to match candidates against
    5. Click "Rank Candidates" to sort by relevance to the job description
    6. View detailed rankings in the "Ranked Results" tab
    7. Download the ranked candidates as an Excel file using the download button in the "Ranked Results" tab
    """)

if __name__ == "__main__":
    main()
