import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_tavily import TavilySearch
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from llama_index.readers.docling import DoclingReader
import dspy
import json

from llama_index.core import Document
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
#from langchain.globals import set_llm_cache
#from langchain_community.cache import SQLiteCache
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import  ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import  ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
import warnings
warnings.filterwarnings('ignore')
# Load environment variables
load_dotenv()

# Initialize the ChatOpenAI instance 
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Set up SQLite cache
#
# 
# set_llm_cache(SQLiteCache(database_path="cache.db"))

# State class for tracking agent's state
class State(TypedDict):
    query: str
    tavily_tool: str
    auto_analyst: dict
    # plot_tool: dict
    # auto_evaluation: dict

# Tavily Search setup
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

def tavily_node(state: State):
    # Initialize Tavily Search Tool
    search = TavilySearchResults(max_results=5)
    tools = [search]
    query = input("Tavily search:")
    agent_executor = [create_react_agent(llm, tools), query]
    response = agent_executor[0].invoke(
        {
            "messages": [HumanMessage(content=query)]
        }
    )
    
    print('1', response['messages'][-1].content)
    return {"tavily_tool": agent_executor[0]}

# DSPy Agents Setup
lm = dspy.LM('openai/gpt-4', model_type='chat', max_tokens=5000)
dspy.configure(lm=lm)


from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import  ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings



Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
Settings.num_output = 512
Settings.context_window = 3900


styling_instructions =[Document(text="""
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
5. Interactive elements:
   - Informative hover text
   - Click-to-hide legend items
"""),

Document(text="""
Line Chart Specific Guidelines:
1. Multiple lines:
   - Use distinct colors with clear contrast
   - Add markers at important points
   - Dashed lines for predicted/estimated values
2. Annotations:
   - Mark global min/max points
   - Highlight significant trend changes
   - Add trend lines where appropriate
3. Range:
   - Consider adding range bands for uncertainty
   - Show reference lines for benchmarks
4. Axis:
   - Date formatting: %Y-%m-%d for daily, %Y-%m for monthly
   - Zero baseline when meaningful
"""),

Document(text="""
Bar Chart Specific Guidelines:
1. Layout:
   - Horizontal bars for long labels
   - Grouped bars: max 4-5 groups
   - Stacked bars: show percentages
2. Annotations:
   - Value labels on the bars
   - Percentage contributions for stacked bars
3. Spacing:
   - Bar gap: 0.2
   - Group gap: 0.3
4. Colors:
   - Single color for simple comparisons
   - Color gradient for ordered data
   - Distinct colors for categories
"""),

Document(text="""
Scatter Plot Specific Guidelines:
1. Markers:
   - Size: vary between 8-12
   - Opacity: 0.7 for density
   - Different symbols for categories
2. Trend lines:
   - Add regression line when relevant
   - Show R-squared value
   - Consider confidence intervals
3. Quadrant analysis:
   - Add mean/median lines
   - Label quadrants if meaningful
4. Bubble plots:
   - Size range: 10-50
   - Add size legend
   - Scale bubbles by area, not radius
"""),

Document(text="""
Distribution Plot Guidelines:
1. Histogram:
   - Optimize bin width
   - Show density curve
   - Mark mean and median
2. Box plots:
   - Show all outlier points
   - Add violin plot overlay option
   - Display sample size
3. KDE plots:
   - Show rug plot
   - Multiple distributions: use transparency
4. Stats:
   - Show basic statistics
   - Add distribution parameters
   - Mark standard deviations
""")]

# Creating an Index
style_index =  VectorStoreIndex.from_documents(styling_instructions) 

# Build query engines over your indexes

                    
csv_reader = DoclingReader()
reader = SimpleDirectoryReader( 
         input_dir="/Users/liatparker/Documents/new agent/",
         file_extractor= {".csv": csv_reader}
         )

documents = reader.load_data()

# Creating an Index
dataframe_index =  VectorStoreIndex.from_documents(documents)
dataframe_engine = dataframe_index.as_query_engine()
#print(dataframe_engine)
styling_engine = style_index.as_query_engine()


# Builds the tools
query_engine_tools = [
    QueryEngineTool(
        query_engine=dataframe_engine,
# Provides the description which helps the agent decide which tool to use 
        metadata=ToolMetadata(
            name="dataframe_index",
            description="Provides information about the data in the data frame",
        ),
\
    ),
    QueryEngineTool(
# Play around with the description to see if it leads to better results
        query_engine=styling_engine,
        metadata=ToolMetadata(
            name="Styling",
            description="Provides instructions on how to style your Plotly plots"
            "Use a detailed plain text question as input to the tool.",
           
        ),
        
    ),
  
]

class analytical_planner(dspy.Signature):
    """You are data analytics planner agent. You have access to three inputs
    1. Full Dataset
    2. Data Agent descriptions
    3. User-defined Goal
    You take these three inputs to develop a comprehensive plan to achieve the user-defined goal from the data & Agents available.
    In case you think the user-defined goal is infeasible you can ask the user to redefine or add more description to the goal.

    Give your output in this format:
    plan: Agent1->Agent2->Agent3
    plan_desc = Use Agent 1 for this reason, then agent2 for this reason and lastly agent3 for this reason.

    You don't have to use all the agents in response of the query
    """
    dataset = dspy.InputField(desc="Available datasets loaded in the system")
    Agent_desc = dspy.InputField(desc="The agents available in the system")
    goal = dspy.InputField(desc="The user defined goal")
    plan = dspy.OutputField(desc="The plan that would achieve the user defined goal")
    plan_desc = dspy.OutputField(desc="The reasoning behind the chosen plan")

class preprocessing_agent(dspy.Signature):
    """You are a data pre-processing agent specialized in unbiased and robust data preparation.
    Your task is to take a user-defined goal and available dataset to build an exploratory analytics pipeline
    that prevents various forms of bias while ensuring data quality.

    BIAS PREVENTION GUIDELINES:
    1. Sampling Bias Prevention:
        - Check for representative sampling across all demographic groups
        - Report sample sizes for different subgroups
        - Use stratified sampling when appropriate
        - Identify and handle class imbalances

    2. Selection Bias Prevention:
        - Document any data filtering criteria
        - Check for missing data patterns across groups
        - Validate data collection methods
        - Report potential selection effects

    3. Measurement Bias Prevention:
        - Standardize measurements across all groups
        - Check for systematic measurement errors
        - Validate data collection instruments
        - Document any proxy variables used

    4. Algorithmic Fairness:
        - Test for disparate impact across protected attributes
        - Use bias-aware preprocessing techniques
        - Document fairness metrics and thresholds
        - Consider multiple fairness criteria

    DATA PREPROCESSING STEPS:
    1. Data Quality Assessment:
        - Check for data completeness
        - Identify outliers and anomalies
        - Validate data types and ranges
        - Document data quality metrics

    2. Missing Data Handling:
        - Analyze missing data patterns
        - Document missing data mechanisms (MCAR, MAR, MNAR)
        - Use appropriate imputation methods
        - Report impact of missing data handling

    3. Feature Engineering:
        - Create unbiased feature representations
        - Document feature creation rationale
        - Test for proxy variables
        - Validate feature importance

    4. Data Transformation:
        - Use robust scaling methods
        - Document transformation rationale
        - Preserve data interpretability
        - Test for transformation impact

    5. Outlier Handling:
        - Use robust statistical methods
        - Document outlier criteria
        - Consider group-specific distributions
        - Preserve important variations

    OUTPUT REQUIREMENTS:
    1. Documentation:
        - Clear comments explaining each step
        - Bias prevention measures taken
        - Data quality metrics
        - Limitations and assumptions

    2. Code Quality:
        - Use pandas and numpy efficiently
        - Include error handling
        - Add validation checks
        - Maintain reproducibility

    3. Reporting:
        - Data quality summary
        - Bias assessment results
        - Processing impact analysis
        - Recommendations for analysis

    You must prevent bias while maintaining data integrity and analytical value.
    """
    dataset = dspy.InputField(desc="Available datasets loaded in the system")
    goal = dspy.InputField(desc="The user defined goal")
    commentary = dspy.OutputField(desc="Detailed explanation of preprocessing steps, bias prevention measures, and their impacts")
    code = dspy.OutputField(desc="Well-documented Python code that implements unbiased preprocessing and analysis")

class statistical_analytics_agent(dspy.Signature):
    """You are a statistical analytics agent. 
    Your task is to take a dataset and a user-defined goal, and output explanation of the data and
    Python code that performs the appropriate statistical analysis to achieve that goal without bias.
    You should use the Python statsmodel library for advanced statistics and pandas for basic statistics.
    
    For full dataset statistics, you should include:
    1. Basic statistics (count, mean, std, min, 25%, 50%, 75%, max) for all numeric columns
    2. Value counts and frequencies for categorical columns
    3. Missing value analysis
    4. Distribution analysis for numeric columns
    5. Correlation analysis between numeric columns
    6. Basic data quality checks
    """
    dataset = dspy.InputField(desc="Available datasets loaded in the system")
    goal = dspy.InputField(desc="The user defined goal for the analysis to be performed")
    commentary = dspy.OutputField(desc="The comments about what analysis is being performed")
    code = dspy.OutputField(desc="The code that does the statistical analysis using statsmodel and pandas")

class sk_learn_agent(dspy.Signature):
    """You are a machine learning agent. 
    Your task is to take a dataset and a user-defined goal, and output Python code that performs comprehensive machine learning analysis.
    For full dataset analysis, you should include:
    1. Feature Analysis:
       - Feature importance analysis
       - Feature correlations
       - Principal Component Analysis (PCA)
       - Feature scaling and normalization
    2. Clustering Analysis:
       - K-means clustering
       - Hierarchical clustering
       - DBSCAN for density-based clustering
    3. Classification/Regression Potential:
       - Basic model fitting for numeric targets
       - Classification metrics for categorical targets
       - Cross-validation scores
    4. Anomaly Detection:
       - Isolation Forest
       - Local Outlier Factor
    
    You should use the scikit-learn library and provide clear explanations of the results."""
    dataset = dspy.InputField(desc="Available datasets loaded in the system")
    goal = dspy.InputField(desc="The user defined goal")
    commentary = dspy.OutputField(desc="Detailed explanation of the ML analysis being performed and its insights")
    code = dspy.OutputField(desc="The code that performs comprehensive ML analysis using scikit-learn")

class code_combiner_agent(dspy.Signature):
    """You are a code combine agent, taking Python code output from many agents and combining the operations into 1 output
    You also fix any errors in the code"""
    agent_code_list = dspy.InputField(desc="A list of code given by each agent")
    refined_complete_code = dspy.OutputField(desc="Refined complete code base")

class data_viz_agent(dspy.Signature):
    """
    You are an advanced AI visualization agent specialized in creating insightful and accurate Plotly visualizations.
    Your primary goals are:
    
    1. Data Understanding:
       - Analyze data types and distributions
       - Identify relationships between variables
       - Handle missing values appropriately
       - Scale and transform data when necessary
    
    2. Visualization Selection:
       - Choose the most appropriate chart type based on:
         * Data types (categorical, numerical, temporal)
         * Number of variables
         * Analysis goal (comparison, distribution, relationship, composition)
       - Consider multiple visualization options when appropriate
    
    3. Best Practices:
       - Ensure visual accuracy and prevent distortion
       - Maintain appropriate aspect ratios
       - Use clear and informative labels
       - Include statistical context when relevant
    
    4. Enhanced Features:
       - Add statistical overlays (trend lines, confidence intervals)
       - Include summary statistics in hover text
       - Provide interactive features (zoom, pan, filters)
       - Use subplots for complex comparisons
    
    5. Error Handling:
       - Validate data before plotting
       - Handle edge cases (outliers, missing data)
       - Provide warnings about potential issues
    
    You must use the tools available to your disposal:
    - dataframe_context: Information about the data structure and content
    - styling_context: Detailed styling guidelines for different chart types
    
    Output clear, well-documented code that creates publication-quality visualizations.
    """
    goal = dspy.InputField(desc="User defined goal which includes information about data and desired visualization")
    dataframe_context = dspy.InputField(desc="Provides detailed information about the data structure, types, and content")
    styling_context = dspy.InputField(desc="Comprehensive styling guidelines for creating professional visualizations")
    code = dspy.OutputField(desc="Well-documented Plotly code that creates an insightful and properly styled visualization")

class goal_refiner_agent(dspy.Signature):
    """You take a user-defined goal given to a AI data analyst planner agent, 
    you make the goal more elaborate using the dataset available and agent_desc"""
    dataset = dspy.InputField(desc="Available dataset loaded in the system")
    Agent_desc = dspy.InputField(desc="The agents available in the system")
    goal = dspy.InputField(desc="The user defined goal")
    refined_goal = dspy.OutputField(desc='Refined goal that helps the planner agent plan better')

class auto_analyst(dspy.Module):
    def __init__(self,agents):
# Defines the available agents, their inputs, and description
        self.agents = {}
        self.agent_inputs ={}
        self.agent_desc =[]
        i =0
        for a in agents:
            name = a.__pydantic_core_schema__['schema']['model_name']
# Using CoT prompting as from experience it helps generate better responses
            self.agents[name] = dspy.ChainOfThought(a)
            self.agent_inputs[name] ={x.strip() for x in str(agents[i].__pydantic_core_schema__['cls']).split('->')[0].split('(')[1].split(',')}
            self.agent_desc.append(str(a.__pydantic_core_schema__['cls']))
            i+=1
# Defining the planner, refine_goal & code combiner agents seperately
# as they don't generate the code & analysis they help in planning, 
# getting better goals & combine the code
        self.planner = dspy.ChainOfThought(analytical_planner)
        self.refine_goal = dspy.ChainOfThought(goal_refiner_agent)
        self.code_combiner_agent = dspy.ChainOfThought(code_combiner_agent)
# these two retrievers are defined using llama-index retrievers
# you can customize this depending on how you want your agents
        self.dataset = dataframe_index.as_retriever(similarity_top_k=10)
        print('hey',self.dataset)
        self.styling_index = style_index.as_retriever(similarity_top_k=5)
        
    def forward(self, query):
# This dict is used to quickly pass arguments for agent inputs
        dict_ ={}
# retrieves the relevant context to the query
        dict_['dataset'] = self.dataset.retrieve(query)[0].text
        if any(phrase in query.lower() for phrase in ["full dataset statistics", "statistics of all the dataset", "full dataset", "sk learn analysis of the full dataset"]):
            # For full dataset analysis, we want to include all available data
            dict_['dataset'] = "\n".join([node.text for node in self.dataset.retrieve(query)])
            print('hello!', dict_['dataset'])
        #print(dict_['dataset'])
        dict_['styling_index'] = self.styling_index.retrieve(query)[0].text
        dict_['goal']=query
        dict_['Agent_desc'] = str(self.agent_desc)
# output_dictionary that stores all agent outputs
        output_dict ={}
# this comes up with the plan
        plan = self.planner(goal =dict_['goal'], dataset=dict_['dataset'], Agent_desc=dict_['Agent_desc'],styling_index = dict_['styling_index'])
        output_dict['analytical_planner'] = plan
        plan_list =[]
        code_list =[]
# if the planner worked as intended it should give agents seperated by ->
        if plan.plan.split('->'):
            plan_list = plan.plan.split('->')
# in case the goal is unclear, it sends it to refined goal agent
        else:
            refined_goal = self.refine_goal(dataset=dict_['dataset'], goal=dict_['goal'], Agent_desc= self.agent_desc)
            self.forward(query=refined_goal)
# passes the goal and other inputs to all respective agents in the plan
        for p in plan_list:
            inputs = {x:dict_[x] for x in self.agent_inputs[p.strip()]}
            output_dict[p.strip()]=self.agents[p.strip()](**inputs)
# creates a list of all the generated code, to be combined as 1 script
            code_list.append(output_dict[p.strip()].code)
# Stores the last output
        output_dict['code_combiner_agent'] = self.code_combiner_agent(agent_code_list = str(code_list))
        
        return output_dict
# you can store all available agent signatures as a list
agents =[preprocessing_agent, statistical_analytics_agent, sk_learn_agent, data_viz_agent]

# Define the agentic system
auto_analyst_system = auto_analyst(agents)    

from llama_index.core import Document
from llama_index.core import VectorStoreIndex





def auto_analyst_node(state: State):
    user_query = input("ask about the dataset:")
    
    auto_analyst_output = auto_analyst_system(query = user_query )   
    print(auto_analyst_output['code_combiner_agent'].values()[1])
    return {"auto_analyst": auto_analyst_output}  


import dspy
from pydantic import BaseModel, Field
# A pydantic validator for the output 


# This defines the signature we would be using for evaluating the total score


from pydantic import BaseModel, Field


class preprocessing_agent(dspy.Signature):
    """You are a data pre-processing agent specialized in unbiased and robust data preparation.
    Your task is to take a user-defined goal and available dataset to build an exploratory analytics pipeline
    that prevents various forms of bias while ensuring data quality.

    BIAS PREVENTION GUIDELINES:
    1. Sampling Bias Prevention:
        - Check for representative sampling across all demographic groups
        - Report sample sizes for different subgroups
        - Use stratified sampling when appropriate
        - Identify and handle class imbalances

    2. Selection Bias Prevention:
        - Document any data filtering criteria
        - Check for missing data patterns across groups
        - Validate data collection methods
        - Report potential selection effects

    3. Measurement Bias Prevention:
        - Standardize measurements across all groups
        - Check for systematic measurement errors
        - Validate data collection instruments
        - Document any proxy variables used

    4. Algorithmic Fairness:
        - Test for disparate impact across protected attributes
        - Use bias-aware preprocessing techniques
        - Document fairness metrics and thresholds
        - Consider multiple fairness criteria

    DATA PREPROCESSING STEPS:
    1. Data Quality Assessment:
        - Check for data completeness
        - Identify outliers and anomalies
        - Validate data types and ranges
        - Document data quality metrics

    2. Missing Data Handling:
        - Analyze missing data patterns
        - Document missing data mechanisms (MCAR, MAR, MNAR)
        - Use appropriate imputation methods
        - Report impact of missing data handling

    3. Feature Engineering:
        - Create unbiased feature representations
        - Document feature creation rationale
        - Test for proxy variables
        - Validate feature importance

    4. Data Transformation:
        - Use robust scaling methods
        - Document transformation rationale
        - Preserve data interpretability
        - Test for transformation impact

    5. Outlier Handling:
        - Use robust statistical methods
        - Document outlier criteria
        - Consider group-specific distributions
        - Preserve important variations

    OUTPUT REQUIREMENTS:
    1. Documentation:
        - Clear comments explaining each step
        - Bias prevention measures taken
        - Data quality metrics
        - Limitations and assumptions

    2. Code Quality:
        - Use pandas and numpy efficiently
        - Include error handling
        - Add validation checks
        - Maintain reproducibility

    3. Reporting:
        - Data quality summary
        - Bias assessment results
        - Processing impact analysis
        - Recommendations for analysis

    You must prevent bias while maintaining data integrity and analytical value.
    """
    dataset = dspy.InputField(desc="Available datasets loaded in the system")
    goal = dspy.InputField(desc="The user defined goal")
    commentary = dspy.OutputField(desc="Detailed explanation of preprocessing steps, bias prevention measures, and their impacts")
    code = dspy.OutputField(desc="Well-documented Python code that implements unbiased preprocessing and analysis")


def main():
     # Test the setup
#     print(response.content)



    workflow = StateGraph(State)

    # Add nodes to the graph
    workflow.add_node("tavily_node", tavily_node)
    workflow.add_node("auto_analyst_output", auto_analyst_node)
    #workflow.add_node("plot_agent", react_plot_node)
    # Add edges to the graph
    workflow.set_entry_point("tavily_node") # Set the entry point of the graph
    workflow.add_edge("tavily_node", "auto_analyst_output")
    #workflow.add_edge("auto_analyst_output", "plot_agent")

    workflow.add_edge("auto_analyst_output", END)
    # Compile the graph
    app = workflow.compile()

    sample_text = """
    Anthropic's MCP (Model Context Protocol) is an open-source powerhouse that lets your applications interact effortlessly with APIs across various systems.
    """
    sample_query = """
    what predicts heart attack?
    """
    # Create the initial state with our sample text
    state_input = {"query": f'{sample_query}'}

    # Run the agent's full workflow on our sample text
    result = app.invoke(state_input)

if __name__ == "__main__":
     main() 
