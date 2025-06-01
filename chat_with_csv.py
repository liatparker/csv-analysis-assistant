import streamlit as st
import os
from typing import TypedDict, List
from dotenv import load_dotenv
import pandas as pd
from langchain_openai import ChatOpenAI
import dspy
import plotly.graph_objects as go
from llama_index.core import Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from agents_for_csv import *
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Streamlit interface setup
st.set_page_config(page_title="CSV Analysis Assistant", layout="wide")
st.title("📊 Intelligent CSV Analysis Assistant")

# Sidebar for file upload and configuration
with st.sidebar:
    st.header("📁 Upload & Configure")
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file:
        st.success("File uploaded successfully!")
    
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API key set!")

# Main content area
if not uploaded_file:
    st.markdown("""
    ### 👋 Welcome to the CSV Analysis Assistant!
    
    Upload your CSV file to get started. I can help you:
    - Analyze patterns and relationships in your data
    - Create insightful visualizations
    - Perform statistical analysis
    - Discover meaningful insights
    """)
    st.stop()

# Load and display the data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

df = load_data(uploaded_file)

# Display data overview
col1, col2, col3 = st.columns([2,1,1])
with col1:
    st.markdown("### 📊 Data Preview")
    st.dataframe(df.head(), use_container_width=True)
with col2:
    st.metric("Rows", df.shape[0])
    st.metric("Columns", df.shape[1])
with col3:
    st.markdown("### 📈 Data Types")
    dtypes = df.dtypes.value_counts()
    for dtype, count in dtypes.items():
        st.metric(f"{dtype}", count)

# Initialize LLM and agents
@st.cache_resource
def initialize_agents():
    # Set up LLM configurations
    Settings.llm = OpenAI(model="gpt-3.5-turbo")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    Settings.num_output = 512
    Settings.context_window = 3900
    
    # Create document from dataframe
    df_text = df.to_string()
    documents = [Document(text=df_text)]
    
    # Create index
    dataframe_index = VectorStoreIndex.from_documents(documents)
    
    return auto_analyst(agents=[preprocessing_agent, statistical_analytics_agent, sk_learn_agent, data_viz_agent])

if api_key:
    try:
        analyst_system = initialize_agents()
        st.success("✨ Analysis system ready!")
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        st.stop()

# Chat interface
st.markdown("### 💬 Ask me about your data")
st.markdown("""
**Example queries:**
- What is the distribution of the data?
- what are the data correlations?
- How to predict ...?
- Create visualizations to understand the data distribution
""")

# Create two columns for chat input
input_col, button_col = st.columns([4,1])
with input_col:
    query = st.text_input(
        "Your query",
        placeholder="Ask me  about your data...",
        label_visibility="collapsed"
    )
with button_col:
    analyze_button = st.button("Analyze", use_container_width=True)

if analyze_button and query and api_key:
    with st.spinner("🤔 Analyzing your data..."):
        try:
            # Get analysis from auto_analyst
            output_dict = analyst_system(query=query)
            
            # Show results in a clean layout
            st.markdown("### 📊 Analysis Results")
            
            # Analysis Plan
            st.markdown("#### Analysis Approach")
            st.info(f"**Plan:** {output_dict['analytical_planner'].plan}")
            st.markdown(f"**Reasoning:** {output_dict['analytical_planner'].plan_desc}")
            
            # Results from each agent
            plan_list = output_dict['analytical_planner'].plan.split('->')
            
            for agent_name in plan_list:
                agent_name = agent_name.strip()
                if agent_name in output_dict:
                    st.markdown(f"#### {agent_name.replace('_', ' ').title()}")
                    
                    # Show commentary
                    st.markdown(output_dict[agent_name].commentary)
                    
                    # Show code in expander
                    with st.expander("View Code"):
                        st.code(output_dict[agent_name].code, language='python')
                    
                    # Handle visualizations
                    if agent_name == 'data_viz_agent':
                        try:
                            local_vars = {'df': df, 'pd': pd, 'px': px, 'go': go}
                            exec(output_dict[agent_name].code, globals(), local_vars)
                            if 'fig' in local_vars:
                                st.plotly_chart(local_vars['fig'], use_container_width=True)
                        except Exception as e:
                            st.error(f"Visualization error: {str(e)}")
            
            # Combined code in expander
            with st.expander("View Complete Analysis Code"):
                st.code(output_dict['code_combiner_agent'].refined_complete_code, language='python')
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

# Add helpful information in sidebar
st.sidebar.markdown("""
### 💡 Tips
- Upload your CSV file first
- Enter your OpenAI API key
- Ask questions in natural language
- Get comprehensive analysis automatically
""")

if __name__ == "__main__":
    pass 
