# Intelligent CSV Analysis Assistant

An AI-powered CSV analysis tool that helps you understand and visualize your data through natural language queries.

## Features

- 📊 Automated data analysis and visualization
- 💬 Natural language interface for data queries
- 📈 Statistical analysis and pattern detection
- 🤖 Machine learning insights
- 🎨 Beautiful visualizations with Plotly

## Requirements

- Python 3.11+
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/csv-analysis-assistant.git
cd csv-analysis-assistant
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run chat_with_csv.py
python agents_for_csv.py
```

2. Upload your CSV file through the web interface
3. Enter your OpenAI API key
4. Start asking questions about your data!

Example queries:
- What is the distribution of the data?
- What are the data correlations?
- How to predict specific outcomes?
- Create visualizations to understand patterns

## Project Structure

```
.
├── __init__.py
├── streamlit_app.py     # Main Streamlit application
├── chat_with_csv.py     # Core analysis logic
└── agents_for_csv.py    # AI agents for analysis
```

