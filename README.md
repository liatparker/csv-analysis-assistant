# Chat with CSV

An intelligent chat interface for analyzing CSV files using multiple specialized AI agents. This project combines various AI agents for data preprocessing, statistical analysis, machine learning, and visualization to provide comprehensive data analysis through natural language interaction.

## Features

- Multiple specialized AI agents:
  - Analytical Planner: Coordinates analysis strategy
  - Preprocessing Agent: Data cleaning and initial analysis
  - Statistical Analytics Agent: Statistical analysis using statsmodels
  - Machine Learning Agent: ML analysis using scikit-learn
  - Data Visualization Agent: Creating plots using Plotly
  - Code Combiner Agent: Integrates code from multiple agents
  - Goal Refiner Agent: Refines user queries for better results

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chat_with_csv.git
cd chat_with_csv
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

1. Place your CSV file in the project directory

2. Run the main script:
```bash
python chat_with_csv.py
```

3. Interact with the system using natural language queries about your data

## Example Queries

- "Analyze the distribution of values in column X"
- "Find correlations between columns A and B"
- "Create a visualization showing the trend of X over time"
- "Build a prediction model for target Y using features A, B, C"

## Project Structure

```
chat_with_csv/
├── chat_with_csv.py     # Main application file
├── requirements.txt     # Project dependencies
├── README.md           # Documentation
└── .env                # API keys and configuration
```

## Requirements

- Python 3.8+
- OpenAI API key
- Tavily API key
- Required Python packages (see requirements.txt)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 