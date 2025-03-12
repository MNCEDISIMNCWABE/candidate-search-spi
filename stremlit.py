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

    # Base clause: search in experience title.
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

    # Additional filter: Company Name (in experience collection)
    if company_filter:
        must_clauses.append({
            "nested": {
                "path": "member_experience_collection",
                "query": {
                    "match_phrase": {
                        "member_experience_collection.company_name": company_filter
                    }
                }
            }
        })

    # Additional filter: University Name (in education collection)
    if university_filter:
        must_clauses.append({
            "nested": {
                "path": "member_education_collection",
                "query": {
                    "match_phrase": {
                        "member_education_collection.title": university_filter
                    }
                }
            }
        })

    # Additional filter: Industry (top-level field)
    if industry_filter:
        must_clauses.append({
            "match_phrase": {
                "industry": industry_filter
            }
        })

    # Additional filter: Skills (in skills collection)
    if skills_filter:
        must_clauses.append({
            "nested": {
                "path": "member_skills_collection",
                "query": {
                    "match_phrase": {
                        "member_skills_collection.member_skill_list.skill": skills_filter
                    }
                }
            }
        })

    # Additional filter: Certifications (in certifications collection)
    if certifications_filter:
        must_clauses.append({
            "nested": {
                "path": "member_certifications_collection",
                "query": {
                    "match_phrase": {
                        "member_certifications_collection.name": certifications_filter
                    }
                }
            }
        })

    # Additional filter: Languages (in languages collection)
    if languages_filter:
        # Convert the search term to lower case so that "English" matches stored "english"
        must_clauses.append({
            "nested": {
                "path": "member_languages_collection",
                "query": {
                    "match_phrase": {
                        "member_languages_collection.member_language_list.language": languages_filter.lower()
                    }
                }
            }
        })

    # Exclude patterns in titles
    exclude_patterns = ["PA to", "Assistant to", "Personal Assistant", "EA to","Executive Assistant to","CFO Designate","CEO Designate"]
    must_not_clauses = [
        {
            "nested": {
                "path": "member_experience_collection",
                "query": {
                    "query_string": {
                        "query": f"member_experience_collection.title:({pattern})",
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
            "term": {
                "country": country_filter
            }
        })

    if location_filter:
        payload["query"]["bool"]["must"].append({
            "match_phrase": {
                "location": location_filter
            }
        })

    # Uncomment for debugging:
    # print(json.dumps(payload, indent=2))

    # Send the search request.
    search_url = "https://api.coresignal.com/cdapi/v1/professional_network/employee/search/es_dsl"
    headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer eyJhbGciOiJFZERTQSIsImtpZCI6IjYzOGY5Y2YyLTUyM2UtOGJmMC0zZmFlLTEyY2UwNTUzOTQ1YiJ9.eyJhdWQiOiJzdHVkZW50LnVqLmFjLnphIiwiZXhwIjoxNzczMjc2NTkzLCJpYXQiOjE3NDE3MTk2NDEsImlzcyI6Imh0dHBzOi8vb3BzLmNvcmVzaWduYWwuY29tOjgzMDAvdjEvaWRlbnRpdHkvb2lkYyIsIm5hbWVzcGFjZSI6InJvb3QiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJzdHVkZW50LnVqLmFjLnphIiwic3ViIjoiOTc4OGQ4OTYtMjcwYy01ODY4LTE2NDItOTFhYmQ5NDBhMDg2IiwidXNlcmluZm8iOnsic2NvcGVzIjoiY2RhcGkifX0.GYI_XfOwh_DiuBMu9q_JRL39v4bOgJixOWIxPG0ZujADWVFtQQKO1tNJ71ig-ncoRJJE7R6z0WbG4Bxjs_qkDw'
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
        collect_url = f"https://api.coresignal.com/cdapi/v1/professional_network/employee/collect/{emp_id}"
        r = requests.get(collect_url, headers=headers)
        r.raise_for_status()
        employee = r.json()

        # Basic fields
        id_val = employee.get("id")
        name_val = employee.get("name")
        headline_val = employee.get("title")
        location_val = employee.get("location")
        country_val = employee.get("country")
        url_val = employee.get("url")
        canonical_url = employee.get("canonical_url")
        industry_val = employee.get("industry")
        experience_count_val = employee.get("experience_count")
        summary_val = employee.get("summary")

        # ----- EXPERIENCE (deduplicate) -----
        raw_exps = employee.get("member_experience_collection", [])
        unique_exps = []
        seen_exps = set()
        for exp in raw_exps:
            key = (
                exp.get("title", "N/A"),
                exp.get("company_name", "N/A"),
                exp.get("date_from", "N/A"),
                exp.get("date_to", "N/A")
            )
            if key not in seen_exps:
                seen_exps.add(key)
                unique_exps.append(exp)
        experiences_str = "\n".join(
            f"Role: {exp.get('title','N/A')} | Company: {exp.get('company_name','N/A')} | From: {exp.get('date_from','N/A')} | To: {exp.get('date_to','N/A')} | Duration: {exp.get('duration','N/A')}"
            for exp in unique_exps
        )

        # ----- EDUCATION (deduplicate) -----
        raw_edu = employee.get("member_education_collection", [])
        unique_edu = []
        seen_edu = set()
        for edu in raw_edu:
            key = (
                edu.get("title", "N/A"),
                edu.get("subtitle", "N/A"),
                edu.get("date_from", "N/A"),
                edu.get("date_to", "N/A")
            )
            if key not in seen_edu:
                seen_edu.add(key)
                unique_edu.append(edu)
        educations_str = "\n".join(
            f"Institution: {edu.get('title','N/A')} | Degree: {edu.get('subtitle','N/A')} | From: {edu.get('date_from','N/A')} | To: {edu.get('date_to','N/A')}"
            for edu in unique_edu
        )

        # ----- SKILLS (deduplicate) -----
        raw_skills = employee.get("member_skills_collection", [])
        seen_skills = set()
        for skill_entry in raw_skills:
            skill_name = skill_entry.get("member_skill_list", {}).get("skill", "N/A")
            seen_skills.add(skill_name)
        skills_str = ", ".join(seen_skills) if seen_skills else ""

        # ----- CERTIFICATIONS (deduplicate) -----
        raw_certifications = employee.get("member_certifications_collection", [])
        seen_certs = set()
        for cert in raw_certifications:
            cert_name = cert.get("name", "N/A")
            seen_certs.add(cert_name)
        certifications_str = ", ".join(seen_certs) if seen_certs else ""

        # ----- LANGUAGES (deduplicate) -----
        raw_languages = employee.get("member_languages_collection", [])
        seen_langs = set()
        for lang in raw_languages:
            language_name = lang.get("member_language_list", {}).get("language", "N/A")
            seen_langs.add(language_name)
        languages_str = ", ".join(seen_langs) if seen_langs else ""

        # ----- PROJECTS (deduplicate) -----
        raw_projects = employee.get("member_projects_collection", [])
        seen_projects = set()
        for proj in raw_projects:
            proj_name = proj.get("name", "N/A")
            seen_projects.add(proj_name)
        projects_str = ", ".join([str(x) for x in seen_projects if x is not None]) if seen_projects else ""

        # Build the final row dictionary.
        row = {
            "ID": id_val,
            "Name": name_val,
            "Headline/Title": headline_val,
            "Location": location_val,
            "Country": country_val,
            "URL": url_val,
            "Canonical_URL": canonical_url,
            "Industry": industry_val,
            "Experience Count": experience_count_val,
            "Summary": summary_val,
            "Experiences": experiences_str,
            "Educations": educations_str,
            "Skills": skills_str,
            "Certifications": certifications_str,
            "Languages": languages_str,
            "Projects": projects_str
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

# Ranking functions
def build_user_text(row, text_columns: List[str]) -> str:
    """
    Combine relevant text fields into a single string for semantic comparison.
    Handles both string and list-type columns.
    
    Args:
        row: DataFrame row containing user information
        text_columns: List of columns to include in combined text
        
    Returns:
        Combined text string
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
    Clean and normalize text input by:
    - Removing emojis and special characters
    - Removing extra whitespace
    - Converting to lowercase
    """
    # Remove emojis using Unicode range patterns
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # CJK symbols
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
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove special characters and punctuation (keep alphanumeric and whitespace)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Convert to lowercase and clean whitespace
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
    Rank candidates based on semantic similarity to job description.
    """
    try:
        logger.info("Starting candidate ranking process...")
        
        # Create working copy to avoid modifying original dataframe
        df = df_employees.copy()
        
        # Set columns for corpus
        if text_columns is None:
            text_columns = ['Summary', 'Experiences', 'Educations', 
                           'Headline/Title', 'Industry', 'Skills'
                           'Certifications','Projects']
            logger.debug(f"Using default text columns: {text_columns}")
        else:
            logger.debug(f"Using custom text columns: {text_columns}")

        # 1) Create combined text for each user
        logger.info("Combining candidate text fields...")
        df['combined_text'] = df.apply(
            lambda x: build_user_text(x, text_columns), 
            axis=1
        )
        logger.info(f"Processed {len(df)} candidate profiles")

        # Handle empty texts to avoid encoding issues
        logger.info("Filtering empty candidate texts...")
        initial_count = len(df)
        df['combined_text'] = df['combined_text'].replace(r'^\s*$', np.nan, regex=True)
        df = df.dropna(subset=['combined_text']).reset_index(drop=True)
        filtered_count = len(df)
        logger.info(f"Removed {initial_count - filtered_count} empty profiles, {filtered_count} remaining")

        if df.empty:
            logger.warning("No valid candidate texts found after preprocessing")
            return pd.DataFrame()

        # 2) Initialize sentence transformer model
        logger.info(f"Initializing sentence transformer model: {model_name}")
        model = SentenceTransformer(model_name)
        
        # 3) Preprocess and embed job description
        logger.info("Preprocessing job description...")
        clean_jd = preprocess_text(job_description)
        logger.debug(f"Job description length: {len(clean_jd.split())} words")
        
        logger.info("Encoding job description...")
        job_embedding = model.encode(clean_jd, convert_to_tensor=True)
        logger.debug(f"Job embedding shape: {job_embedding.shape}")

        # 4) Embed candidate texts in batches
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

        # 5) Calculate cosine similarities
        logger.info("Calculating cosine similarities...")
        similarities = util.cos_sim(job_embedding, user_embeddings)
        df['similarity_score'] = similarities.cpu().numpy().flatten()
        
        # Calculate score statistics
        min_score = df['similarity_score'].min()
        max_score = df['similarity_score'].max()
        logger.info(f"Similarity scores range: {min_score:.3f} - {max_score:.3f}")
        logger.debug(f"Score distribution:\n{df['similarity_score'].describe()}")

        # 6) Sort and return results
        logger.info("Sorting candidates by similarity score...")
        df_sorted = df.sort_values(by='similarity_score', ascending=False)\
                      .reset_index(drop=True)
        
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
    """Display the login page and handle authentication."""
    st.title("SPI Executive Search ")
    
    # Create login and signup tabs
    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
    
    with login_tab:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            users = load_users()
            
            if username in users and check_password(users[username]['password'], password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.user_role = users[username].get('role', 'user')
                st.success(f"Welcome back, {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with signup_tab:
        if st.session_state.get('user_role') == 'admin' or not os.path.exists('users.pkl'):
            new_username = st.text_input("New Username", key="new_username")
            new_password = st.text_input("New Password", type="password", key="new_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
            email = st.text_input("Email", key="email")
            
            if st.button("Sign Up"):
                users = load_users()
                
                if new_username in users:
                    st.error("Username already exists")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                elif not new_username or not new_password:
                    st.error("Username and password cannot be empty")
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
            st.info("User registration is currently managed by administrators.")

def logout():
    """Handle user logout."""
    if st.sidebar.button("Logout"):
        for key in ['logged_in', 'username', 'user_role']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Admin dashboard functions
def admin_dashboard():
    """Display the admin dashboard for user management."""
    st.title("Admin Dashboard - User Management")
    
    users = load_users()
    
    # Display list of users
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
    
    # User management sections
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
    
    # Delete user section
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

# Main application function
def main():
    st.set_page_config(page_title="Candidate Search & Match", layout="wide")
    
    # Initialize session state variables
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'ranked_results' not in st.session_state:
        st.session_state.ranked_results = None
    
    # Check if users file exists, if not create it with default admin
    if not os.path.exists('users.pkl'):
        load_users()
    
    # Display sidebar for logged-in users
    if st.session_state.logged_in:
        st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
        st.sidebar.write(f"Role: **{st.session_state.user_role}**")
        logout()
        
        # Admin navigation
        if st.session_state.user_role == 'admin':
            pages = ["Candidate Search", "Admin Dashboard"]
            selected_page = st.sidebar.selectbox("Navigation", pages)
            
            if selected_page == "Admin Dashboard":
                admin_dashboard()
                return
    
    # Display login page if not logged in
    if not st.session_state.logged_in:
        login_page()
        return
    
    # Main application - only shown to logged in users
    st.title("Candidate Search & Match")
    st.markdown("Find and rank the best candidates for a job position")
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Search Candidates", "Ranked Results"])
    
    with tab1:
        st.header("Search for Candidates")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Search inputs - Horizontal layout
            st.subheader("Search Criteria")
            
            # Job Title
            search_query = st.text_input("Job Title/Position", 
                                       placeholder="e.g. '(Chief Financial Officer) OR (CFO)'")
            
            # Location filters - Horizontal
            loc_col1, loc_col2 = st.columns(2)
            with loc_col1:
                country = st.text_input("Country", placeholder="e.g. South Africa")
            with loc_col2:
                location = st.text_input("City", placeholder="e.g. Johannesburg")
            
            # Company/University - Horizontal
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                company_filter = st.text_input("Company Name", placeholder="e.g. PWC")
            with comp_col2:
                university_filter = st.text_input("University Name", placeholder="e.g. University of Cape Town")
            
            # Industry/Skills - Horizontal
            industry_col1, industry_col2 = st.columns(2)
            with industry_col1:
                industry_filter = st.text_input("Industry", placeholder="e.g. Accounting")
            with industry_col2:
                skills_filter = st.text_input("Skills", placeholder="e.g. business strategy, financial modeling")
            
            # Certifications/Languages - Horizontal
            cert_col1, cert_col2 = st.columns(2)
            with cert_col1:
                certifications_filter = st.text_input("Certifications", placeholder="e.g. Assessor")
            with cert_col2:
                languages_filter = st.text_input("Languages", placeholder="e.g. English")
            
            # Max results and Search button - Horizontal
            slider_col, btn_col = st.columns([2, 1])
            with slider_col:
                max_results = st.slider("Maximum number of results", 1, 400, 15)
            with btn_col:
                st.write("")  # Spacer
                st.write("")  # Spacer
                search_button = st.button("Search Candidates")
            
            if search_button and search_query:
                with st.spinner("Searching for candidates..."):
                    # Clear previous results
                    st.session_state.ranked_results = None
                    
                    # Perform search with all filters
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
            # Job description for ranking
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
                        # Load model (uses cache)
                        load_model()
                        
                        # Rank candidates
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
        
        # Display search results
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
                    
                    # Display experience details
                    if pd.notnull(row['Experiences']) and row['Experiences']:
                        st.markdown("---")
                        st.markdown("### Experience Details")
                        experiences = row['Experiences'].split('\n')
                        for exp in experiences:
                            st.markdown(f"- {exp}")
                    
                    # Display education details
                    if pd.notnull(row['Educations']) and row['Educations']:
                        st.markdown("---")
                        st.markdown("### Education Details")
                        educations = row['Educations'].split('\n')
                        for edu in educations:
                            st.markdown(f"- {edu}")

    with tab2:
        st.header("Ranked Candidates")
        
        if st.session_state.ranked_results is not None and not st.session_state.ranked_results.empty:
            # Add download button for Excel export
            export_columns = [
                'ID', 'Name', 'Headline/Title', 'Location', 'Country', 'URL', 
                'Industry', 'Experience Count', 'Summary', 'Experiences', 
                'Educations', 'Skills', 'combined_text', 'similarity_score'
            ]
            
            # Create a copy with only the requested columns
            export_df = st.session_state.ranked_results[
                [col for col in export_columns if col in st.session_state.ranked_results.columns]
            ].copy()
            
            # Convert similarity score to percentage for better readability in Excel
            if 'similarity_score' in export_df.columns:
                export_df['similarity_score'] = export_df['similarity_score'] * 100
            
            excel_data = to_excel(export_df)
            
            st.download_button(
                label="📥 Download Ranked Candidates (Excel)",
                data=excel_data,
                file_name='ranked_candidates.xlsx',
                mime='application/vnd.ms-excel',
            )
            
            # Show ranking summary
            st.subheader("Match Results")
            
            # Create a bar chart of match percentages
            top_candidates = st.session_state.ranked_results.head(10)
            chart_data = pd.DataFrame({
                'Candidate': top_candidates['Name'],
                'Match Percentage': top_candidates['similarity_score'] * 100
            })
            
            st.bar_chart(chart_data.set_index('Candidate'))
            
            # Show ranked candidates
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
                    
                    # Display experience details WITHOUT nested expanders
                    if pd.notnull(row['Experiences']) and row['Experiences']:
                        st.markdown("---")
                        st.markdown("### Experience Details")
                        experiences = row['Experiences'].split('\n')
                        for exp in experiences:
                            st.markdown(f"- {exp}")
                    
                    # Display education details WITHOUT nested expanders
                    if pd.notnull(row['Educations']) and row['Educations']:
                        st.markdown("---")
                        st.markdown("### Education Details")
                        educations = row['Educations'].split('\n')
                        for edu in educations:
                            st.markdown(f"- {edu}")
        else:
            st.info("No ranked results available. Please search for candidates and rank them first.")

    # Add footer with instructions
    st.markdown("---")
    st.markdown("""
    **How to use this application:**
    1. Enter a job title/position query in the search box (use OR for multiple terms)
    2. Use advanced filters to narrow down candidates
    3. Click "Search Candidates" to find matching profiles
    4. Enter a detailed job description to match candidates against
    5. Click "Rank Candidates" to sort by relevance to the job description
    6. View detailed rankings in the "Ranked Results" tab
    7. Download the ranked candidates as an Excel file using the download button
    """)

if __name__ == "__main__":
    main()
