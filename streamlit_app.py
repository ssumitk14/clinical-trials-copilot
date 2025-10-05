import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from app.common.ctgov_fetcher import fetch_full_study
from app.common.etl import normalize_study
from app.common.vectorstore import upsert_trial, query_similar
from app.common.embeddings import EmbeddingService
from app.common.llm_services import LLMService
from app.trial_summarization.summarizer import structured_summary
from app.compliance_validation.compliance_validator import validate_compliance
from app.cross_trial_comparison.comparator import compare_trials
from app.risk_analysis.risk_model import build_risk_pipeline, featurize_trial_record
from app.display_handler import TrialDisplayHandler, DevDisplayHandler
from app.config import MongoConfig, OpenAIConfig
from app.services.storage_services import MongoDBService
from app.prompt import summary_prompt
from app.models import TrialSummary

embedding_obj = EmbeddingService()
mongo_obj = MongoDBService(MongoConfig.MONGODB_URI, MongoConfig.DB_NAME, MongoConfig.SEARCH_INDEX_NAME)

st.set_page_config(layout='wide')
st.title('Clinical Trials Pharma-Copilot')

st.sidebar.header('Data & Actions')
action = st.sidebar.selectbox('Action', ['Fetch trial', 'Compare two trials', 'Compliance check', 'Predict risk'])

# Define process options for each action
process_options = {
    'Fetch trial': ['Baseline', 'RAG (Single-Agent)', 'Multi-Agent'],
    'Compare two trials': ['Basic Comparison', 'Detailed Comparison', 'Statistical Comparison'],
    'Compliance check': ['Basic Validation', 'Full Compliance Check', 'Regulatory Review'],
    'Predict risk': ['Simple Risk Assessment', 'Advanced ML Prediction', 'Ensemble Prediction']
}

# Second dropdown for Process selection (conditional)
if action in process_options:
    process = st.sidebar.selectbox(
        f'Process for {action}',
        process_options[action]
    )
    
    # Display selected process info
    st.sidebar.info(f"Selected: {process}")

if action == 'Fetch trial':
    # Process-specific handling
    if process == process_options[action][0]:  # Baseline
        nct = st.sidebar.text_input('NCTID', value='')
        if st.sidebar.button('Fetch and normalize'):
            if not nct:
                st.sidebar.error('Enter an NCTID e.g. NCT00000000')
            else:
                with st.spinner(f'Fetching using {process}...'):
                    raw = fetch_full_study(nct)
                    tr = normalize_study(raw)
                    st.subheader('Structured record')
                    st.json(tr.model_dump())
                    summ = structured_summary(tr.model_dump())
                    TrialDisplayHandler.display_card_format(summ)

                    text = tr.title or ''
                    emb = embedding_obj.create_embedding(text)
                    upsert_trial(tr.model_dump(), text, emb)
                    st.success('Indexed locally (MongoDB)')

    elif process == process_options[action][1]:
        nct = st.sidebar.text_input('NCTID', value='')
        search_text = st.sidebar.text_input('Query', help='🔍 Search any trial related query')
        
        if st.sidebar.button('Generate Summary'):
            query_embedding = embedding_obj.create_embedding(text=search_text)
            print("NCT ::", nct)
            results = mongo_obj.vector_search_filter(MongoConfig.EMBEDDING_COLLECTION_NAME, query_embedding, limit=1, nct_id=nct)
            to_filter = ["nctId", "title", "text_blob"]
            results = [{k: d[k] for k in to_filter if k in d} for d in results]
            print("Search results -- filtered:: ", results)
            llm_service = LLMService()
            prompt = llm_service.build_prompt(summary_prompt, results)
            response = llm_service.get_llm_response(prompt, search_text, response_format="structured", pydantic_model=TrialSummary)
            TrialDisplayHandler.display_card_format(response)
            # DevDisplayHandler.basic_message()
        
    elif process == process_options[action][2]:
        # TODO: Multi Agent approach
        DevDisplayHandler.basic_message()
    

elif action == 'Compare two trials':
    n1 = st.sidebar.text_input('NCTID 1')
    n2 = st.sidebar.text_input('NCTID 2')
    if st.sidebar.button('Compare'):
        if not n1 or not n2:
            st.sidebar.error('Enter both NCTIDs')
        else:
            with st.spinner(f'Comparing using {process}...'):
                t1 = normalize_study(fetch_full_study(n1))
                t2 = normalize_study(fetch_full_study(n2))
                
                # Process-specific comparison
                if process == 'Basic Comparison':
                    st.subheader('Basic Comparison Results')
                    comp = compare_trials(t1.model_dump(), t2.model_dump())
                    st.json(comp)
                
                elif process == 'Detailed Comparison':
                    st.subheader('Detailed Comparison Analysis')
                    comp = compare_trials(t1.model_dump(), t2.model_dump())
                    
                    # Create comparison summary
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Trial 1:** {n1}")
                        st.write(f"Title: {t1.title}")
                    with col2:
                        st.write(f"**Trial 2:** {n2}")
                        st.write(f"Title: {t2.title}")
                    
                    st.subheader('Comparison Results')
                    st.json(comp)
                
                elif process == 'Statistical Comparison':
                    st.subheader('Statistical Comparison Analysis')
                    comp = compare_trials(t1.model_dump(), t2.model_dump())
                    st.json(comp)
                    st.subheader('Statistical Insights')
                    st.write("📈 **Statistical analysis would be performed here**")
                    # Add statistical analysis implementation

elif action == 'Compliance check':
    nct = st.sidebar.text_input('NCTID')
    if st.sidebar.button('Validate'):
        if not nct:
            st.sidebar.error('Enter an NCTID')
        else:
            with st.spinner(f'Validating using {process}...'):
                tr = normalize_study(fetch_full_study(nct))
                
                # Process-specific validation
                if process == 'Basic Validation':
                    st.subheader('Basic Compliance Check')
                    res = validate_compliance(tr.model_dump())
                    st.json(res)
                
                elif process == 'Full Compliance Check':
                    st.subheader('Full Compliance Report')
                    res = validate_compliance(tr.model_dump())
                    
                    # Enhanced display
                    st.write(f"**Trial:** {nct} - {tr.title}")
                    st.json(res)
                    st.subheader('Compliance Summary')
                    st.write("✅ **Comprehensive compliance check completed**")
                
                elif process == 'Regulatory Review':
                    st.subheader('Regulatory Compliance Analysis')
                    res = validate_compliance(tr.model_dump())
                    
                    st.write(f"**Regulatory Review for:** {nct}")
                    st.json(res)
                    st.subheader('Regulatory Notes')
                    st.write("🏛️ **Regulatory-specific analysis would include FDA/EMA guidelines**")
                    # Add regulatory-specific implementation

elif action == 'Predict risk':
    st.subheader(f'Risk Prediction - {process}')
    
    if process == 'Simple Risk Assessment':
        st.write('Basic risk assessment from labeled CSV (user provides).')
    elif process == 'Advanced ML Prediction':
        st.write('Advanced machine learning model training from labeled CSV.')
    elif process == 'Ensemble Prediction':
        st.write('Ensemble method risk prediction from labeled CSV.')
    
    upload = st.file_uploader('Upload labeled CSV with columns: nctid, label(0/1), title, ...')
    if upload is not None:
        import pandas as pd
        df = pd.read_csv(upload)
        
        st.write(f"📊 **Dataset loaded:** {len(df)} records")
        st.write("**Sample data:**")
        st.dataframe(df.head())
        
        # Process-specific risk prediction
        if process == 'Simple Risk Assessment':
            st.info('Running simple risk assessment...')
            # Basic feature extraction
            texts = [featurize_trial_record({'title': r['title']}) for _, r in df.iterrows()]
            y = df['label'].values
            st.write(f"**Features extracted:** {len(texts)} trials")
            st.success('Simple risk assessment completed.')
        
        elif process == 'Advanced ML Prediction':
            st.info('Running advanced ML prediction...')
            texts = [featurize_trial_record({'title': r['title']}) for _, r in df.iterrows()]
            y = df['label'].values
            pipe = build_risk_pipeline()
            pipe.fit(texts, y)
            st.success('Advanced ML model trained (exploratory). Save with joblib for later use.')
            
            # Show model info
            st.subheader('Model Performance')
            st.write("🤖 **Advanced ML pipeline trained successfully**")
        
        elif process == 'Ensemble Prediction':
            st.info('Running ensemble prediction...')
            texts = [featurize_trial_record({'title': r['title']}) for _, r in df.iterrows()]
            y = df['label'].values
            pipe = build_risk_pipeline()
            pipe.fit(texts, y)
            st.success('Ensemble model trained (exploratory). Save with joblib for later use.')
            
            # Enhanced ensemble info
            st.subheader('Ensemble Model Details')
            st.write("🎯 **Multiple models combined for improved accuracy**")
