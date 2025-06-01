import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import dspy
from llama_index.core import VectorStoreIndex, Document, SimpleDirectoryReader
from llama_index.readers.docling import DoclingReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Initialize settings
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Streamlit page config
st.set_page_config(
    page_title="CSV Analysis & Visualization Assistant",
    page_icon="📊",
    layout="wide"
)

# Sidebar for file upload and settings
with st.sidebar:
    st.title("📊 Data Analysis Settings")
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file:
        # Save the uploaded file temporarily
        with open("temp.csv", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Read the CSV file
        df = pd.read_csv("temp.csv")
        st.success("File uploaded successfully!")
        
        # Display basic dataset info
        st.write("Dataset Info:")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")
        
        # Initialize DSPy LLM
        lm = dspy.LM('openai/gpt-4', model_type='chat', max_tokens=5000)
        dspy.configure(lm=lm)

def initialize_agents():
    """Initialize the necessary agents and indexes"""
    # Load styling instructions
    styling_instructions = [Document(text="""
    General Styling Guidelines for All Plots:
    1. Always use plotly_white template as the base
    2. Consistent styling:
       - Axes lines width: 0.2
       - Grid width: 1
       - Font: Arial for better readability
       - Title: Bold HTML tags (<b>), positioned at center
       - Axis labels: Bold HTML tags, clear units in parentheses
    3. Number formatting:
       - Use K for thousands (e.g., 10K)
       - Use M for millions (e.g., 2.5M)
       - Percentages with 2 decimal places
    4. Color schemes:
       - Use colorblind-friendly palettes
       - Sequential data: Blues or Viridis
       - Categorical data: Qualitative Set 2
    """)]
    
    # Create indexes
    style_index = VectorStoreIndex.from_documents(styling_instructions)
    
    if os.path.exists("temp.csv"):
        csv_reader = DoclingReader()
        reader = SimpleDirectoryReader(
            input_files=["temp.csv"],
            file_extractor={".csv": csv_reader}
        )
        documents = reader.load_data()
        dataframe_index = VectorStoreIndex.from_documents(documents)
        return style_index, dataframe_index
    return None, None

def main():
    st.title("📊 Intelligent CSV Analysis Assistant")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'indexes_initialized' not in st.session_state:
        st.session_state.indexes_initialized = False
    
    # Main interface
    uploaded_file = st.file_uploader("📂 Upload your CSV file", type=['csv'])
    
    if uploaded_file:
        # Save and process the file
        with open("temp.csv", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Read the CSV file
        df = pd.read_csv("temp.csv")
        
        # Initialize indexes if needed
        if not st.session_state.indexes_initialized:
            style_index, dataframe_index = initialize_agents()
            if style_index and dataframe_index:
                st.session_state.style_index = style_index
                st.session_state.dataframe_index = dataframe_index
                st.session_state.indexes_initialized = True
                
                # Initialize DSPy LLM
                lm = dspy.LM('openai/gpt-4', model_type='chat', max_tokens=5000)
                dspy.configure(lm=lm)
        
        # Show basic dataset info in a cleaner format
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Rows", df.shape[0])
        with col2:
            st.metric("Number of Columns", df.shape[1])
        
        # Initialize auto_analyst system
        from chat_with_csv import preprocessing_agent, statistical_analytics_agent, sk_learn_agent, data_viz_agent, auto_analyst
        agents = [preprocessing_agent, statistical_analytics_agent, sk_learn_agent, data_viz_agent]
        auto_analyst_system = auto_analyst(agents)
        
        # Chat interface
        st.markdown("### 💬 Ask anything about your data")
        st.markdown("""
        **Example queries:**
        - What are the main patterns in this dataset?
        - How are different variables related to each other?
        - What insights can you find about [specific column]?
        - Create visualizations to understand the data distribution
        """)
        
        # Create two columns for chat input
        input_col, button_col = st.columns([4,1])
        with input_col:
            query = st.text_input(
                "Your query",
                placeholder="Ask me anything about your data...",
                label_visibility="collapsed"
            )
        with button_col:
            analyze_button = st.button("Analyze", use_container_width=True)
        
        if analyze_button and query:
            with st.spinner("🤔 Analyzing your data..."):
                try:
                    # Get analysis from auto_analyst
                    output_dict = auto_analyst_system(query=query)
                    
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
                    
                    # Store interaction
                    st.session_state.messages.append({"role": "user", "content": query})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Analysis: {output_dict['analytical_planner'].plan}"
                    })
                    
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
        
        # Show chat history in a clean format
        if st.session_state.messages:
            with st.expander("Previous Queries"):
                for idx, message in enumerate(st.session_state.messages):
                    if message["role"] == "user":
                        st.markdown(f"**Q{idx//2 + 1}:** {message['content']}")
                    else:
                        st.markdown(f"**A{idx//2 + 1}:** {message['content']}")
                        st.divider()
    
    else:
        # Welcome message when no file is uploaded
        st.markdown("""
        ### 👋 Welcome to the CSV Analysis Assistant!
        
        Upload your CSV file to get started. I can help you:
        - Analyze patterns and relationships in your data
        - Create insightful visualizations
        - Perform statistical analysis
        - Discover meaningful insights
        """)

if __name__ == "__main__":
    main() 