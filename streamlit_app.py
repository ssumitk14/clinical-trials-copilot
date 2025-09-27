import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.common.ctgov_fetcher import fetch_full_study
from app.common.etl import normalize_study
from app.common.vectorstore import upsert_trial, query_similar
from app.trial_summarization.summarizer import structured_summary
from app.compliance_validation.compliance_validator import validate_compliance
from app.cross_trial_comparison.comparator import compare_trials
from app.risk_analysis.risk_model import build_risk_pipeline, featurize_trial_record
from app.common.embeddings import EmbeddingService

embedding_obj = EmbeddingService()

st.set_page_config(layout='wide')
st.title('Clinical Trials Pharma-Copilot')

st.sidebar.header('Data & Actions')
action = st.sidebar.selectbox('Action', ['Fetch trial', 'Compare two trials', 'Compliance check', 'Predict risk'])

if action == 'Fetch trial':
    nct = st.sidebar.text_input('NCTID', value='')
    if st.sidebar.button('Fetch and normalize'):
        if not nct:
            st.sidebar.error('Enter an NCTID e.g. NCT00000000')
        else:
            with st.spinner('Fetching...'):
                raw = fetch_full_study(nct)
                tr = normalize_study(raw)
                st.subheader('Structured record')
                st.json(tr.model_dump())
                st.subheader('Structured summary (LLM)')
                summ = structured_summary(tr.model_dump())
                st.json(summ)
                # index into vectorstore
                text = tr.title or ''
                emb = embedding_obj.create_embedding(text)
                upsert_trial(tr.model_dump(), text, emb)
                st.success('Indexed locally (MongoDB)')

elif action == 'Compare two trials':
    n1 = st.sidebar.text_input('NCTID 1')
    n2 = st.sidebar.text_input('NCTID 2')
    if st.sidebar.button('Compare'):
        t1 = normalize_study(fetch_full_study(n1))
        t2 = normalize_study(fetch_full_study(n2))
        comp = compare_trials(t1.model_dump(), t2.model_dump())
        st.json(comp)

elif action == 'Compliance check':
    nct = st.sidebar.text_input('NCTID')
    if st.sidebar.button('Validate'):
        tr = normalize_study(fetch_full_study(nct))
        res = validate_compliance(tr.model_dump())
        st.json(res)

elif action == 'Predict risk':
    st.write('Exploratory model training from labeled CSV (user provides).')
    upload = st.file_uploader('Upload labeled CSV with columns: nctid, label(0/1), title, ...')
    if upload is not None:
        import pandas as pd
        df = pd.read_csv(upload)
        texts = [featurize_trial_record({'title': r['title']}) for _, r in df.iterrows()]
        y = df['label'].values
        pipe = build_risk_pipeline()
        pipe.fit(texts, y)
        st.success('Model trained (exploratory). Save with joblib for later use.')
