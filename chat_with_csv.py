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
import dspy
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

# Load environment variables
load_dotenv()

# Initialize the ChatOpenAI instance 
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Set up SQLite cache
set_llm_cache(SQLiteCache(database_path="cache.db"))

# State class for tracking agent's state
class State(TypedDict):
    query: str
    tavily_tool: str
    auto_analyst: dict
    plot_tool: dict
    auto_evaluation: dict

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
lm = dspy.LM('openai/gpt-4', model_type='chat', max_tokens=1000)
dspy.configure(lm=lm)

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
    """You are a data pre-processing agent, your job is to take a user-defined goal and available dataset,
    to build an exploratory analytics pipeline. You do this by outputing the required Python code. 
    You will only use numpy and pandas, to perform pre-processing and introductory analysis according to data types.
    you must prevent bias.
    """
    dataset = dspy.InputField(desc="Available datasets loaded in the system")
    goal = dspy.InputField(desc="The user defined goal")
    commentary = dspy.OutputField(desc="The comments about what analysis is being performed")
    code = dspy.OutputField(desc="The code that does the data preprocessing prevent bias conclusions and introductory analysis")

class statistical_analytics_agent(dspy.Signature):
    """You are a statistical analytics agent. 
    Your task is to take a dataset and a user-defined goal, and output 
    Python code that performs the appropriate statistical analysis to achieve that goal without bias.
    You should use the Python statsmodel library"""
    dataset = dspy.InputField(desc="Available datasets loaded in the system")
    goal = dspy.InputField(desc="The user defined goal for the analysis to be performed")
    commentary = dspy.OutputField(desc="The comments about what analysis is being performed")
    code = dspy.OutputField(desc="The code that does the statistical analysis using statsmodel")

class sk_learn_agent(dspy.Signature):
    """You are a machine learning agent. 
    Your task is to take a dataset and a user-defined goal, and output Python code that performs the appropriate machine learning analysis to achieve that goal. 
    You should use the scikit-learn library."""
    dataset = dspy.InputField(desc="Available datasets loaded in the system")
    goal = dspy.InputField(desc="The user defined goal")
    commentary = dspy.OutputField(desc="The comments about what analysis is being performed")
    code = dspy.OutputField(desc="The code that does the Exploratory data analysis")

class code_combiner_agent(dspy.Signature):
    """You are a code combine agent, taking Python code output from many agents and combining the operations into 1 output
    You also fix any errors in the code"""
    agent_code_list = dspy.InputField(desc="A list of code given by each agent")
    refined_complete_code = dspy.OutputField(desc="Refined complete code base")

class data_viz_agent(dspy.Signature):
    """
    You are AI agent who uses the goal to generate data visualizations in Plotly.
    You have to use the tools available to your disposal
    {dataframe_index}
    {styling_index}

    You must give an output as code, in case there is no relevant columns, just state that you don't have the relevant information
    """
    goal = dspy.InputField(desc="user defined goal which includes information about data and chart they want to plot")
    dataframe_context = dspy.InputField(desc="Provides information about the data in the data frame.")
    styling_context = dspy.InputField(desc='Provides instructions on how to style your Plotly plots')
    code = dspy.OutputField(desc="Plotly code that visualizes what the user needs according to the query & dataframe_index & styling_context")

class goal_refiner_agent(dspy.Signature):
    """You take a user-defined goal given to a AI data analyst planner agent, 
    you make the goal more elaborate using the dataset available and agent_desc"""
    dataset = dspy.InputField(desc="Available dataset loaded in the system")
    Agent_desc = dspy.InputField(desc="The agents available in the system")
    goal = dspy.InputField(desc="The user defined goal")
    refined_goal = dspy.OutputField(desc='Refined goal that helps the planner agent plan better')

def main():
    # Test the setup
    response = llm.invoke("Hello! Are you working?")
    print(response.content)

if __name__ == "__main__":
    main() 