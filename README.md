# AlphaAgents: Multi-Agent Equity Analysis System

A sophisticated multi-agent AI system for stock analysis that implements collaborative decision-making through agent debate and consensus mechanisms. Built with real-time financial data integration and an intuitive web interface.

## Overview

AlphaAgents is an advanced equity analysis system that employs three specialized AI agents to evaluate stocks from different perspectives. The system uses collaborative decision-making where agents can debate their findings and reach consensus, or fall back to majority voting when agreement cannot be reached.

## Key Features

### Three Specialized AI Agents
- **Fundamental Agent**: Analyzes financial statements, company fundamentals, and long-term value assessment
- **Sentiment Agent**: Processes financial news, market sentiment, and short-term impact factors  
- **Valuation Agent**: Examines price trends, technical indicators, and market-based valuation metrics

### Advanced Decision Making Process
- **Multi-Round Debate**: Up to 5 rounds of AI-powered agent discussions
- **Consensus Detection**: Automatic agreement identification at each stage
- **Majority Voting**: Democratic fallback mechanism when consensus isn't reached
- **Tie Handling**: Conservative HOLD default for uncertain scenarios

### Real-Time Data Integration
- **Financial Data**: Live market data via yfinance API
- **News Analysis**: Current sentiment analysis via NewsAPI
- **AI Reasoning**: Powered by OpenAI GPT-3.5-turbo
- **Rate Limiting**: Smart API usage management and error handling

### Web Interface
- **Streamlit Dashboard**: Clean, interactive web-based user interface
- **Real-Time Progress**: Live analysis tracking with progress indicators
- **Debate Visualization**: Complete agent conversation history and reasoning
- **Result Breakdown**: Detailed voting statistics and confidence metrics

## Technical Stack

- **Backend**: Python 3.9+
- **AI/ML**: OpenAI GPT-3.5-turbo for agent reasoning
- **Financial Data**: yfinance for market data, NewsAPI for sentiment
- **Web Framework**: Streamlit for user interface
- **Data Processing**: pandas, numpy for data manipulation
- **Logging**: Comprehensive system monitoring and debugging

## Installation

### Prerequisites
- Python 3.9 or higher
- OpenAI API key
- NewsAPI key (optional, for sentiment analysis)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/codenlogic78/AlphaAgents.git
   cd AlphaAgents
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   NEWS_API_KEY=your_newsapi_key_here
   API_DELAY=2
   MAX_RETRIES=1
   OPENAI_MAX_CALLS_PER_ANALYSIS=3
   OPENAI_CALL_DELAY=2
   ```

## Usage

### Command Line Interface

Run the system directly from the terminal:
```bash
python main.py
```

Follow the prompts to:
- Enter stock symbol(s) (comma-separated for multiple)
- Select risk tolerance (conservative/neutral/aggressive)
- View real-time analysis results

### Web Interface

Launch the Streamlit web application:
```bash
streamlit run streamlit_simple.py
```

Access the interface at `http://localhost:8501` to:
- Input stock symbols and risk preferences
- Monitor real-time analysis progress
- View detailed agent debates and reasoning
- Examine final recommendations with confidence levels

## System Architecture

### Decision Flow
1. **Individual Analysis**: Each agent analyzes the stock independently
2. **Consensus Check**: System checks if all agents agree immediately
3. **Multi-Round Debate**: If disagreement exists, agents engage in structured debate
4. **Consensus Validation**: After each debate round, system checks for agreement
5. **Majority Voting**: If no consensus after maximum rounds, majority vote determines outcome
6. **Result Generation**: Comprehensive analysis report with method transparency

### Agent Specializations
- **Fundamental Agent**: Focuses on financial health, earnings, growth metrics
- **Sentiment Agent**: Analyzes news sentiment, market psychology, short-term catalysts
- **Valuation Agent**: Evaluates technical indicators, price trends, market timing

## Example Output

### Immediate Consensus Example
```
Analyzing AAPL...
├── Fundamental Agent: BUY (Strong financials, consistent growth)
├── Sentiment Agent: BUY (Positive news coverage, analyst upgrades)  
├── Valuation Agent: BUY (Technical indicators show bullish trend)
└── Result: Immediate Consensus → BUY (High Confidence)
```

### Debate Consensus Example
```
Analyzing MRNA...
├── Initial Positions: F:BUY, S:HOLD, V:BUY (No immediate consensus)
├── Round 1 Debate: Agents discuss fundamental vs sentiment concerns
├── Round 2 Debate: Focus shifts to valuation and risk factors
├── Round 3 Debate: Consensus reached on conservative approach
└── Result: Debate Consensus → HOLD (High Confidence)
```

## Decision Methods

The system employs four distinct decision-making methods:

1. **Immediate Consensus**: All agents agree on initial analysis
2. **Debate Consensus**: Agreement reached through multi-round discussion
3. **Majority Voting**: Democratic decision when debate fails to reach consensus
4. **Tie Default**: Conservative HOLD recommendation for tied votes

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for AI agent reasoning
- `NEWS_API_KEY`: Optional for enhanced sentiment analysis
- `API_DELAY`: Delay between financial API calls (default: 2 seconds)
- `MAX_RETRIES`: Maximum retries for failed API calls (default: 1)
- `OPENAI_MAX_CALLS_PER_ANALYSIS`: Limit OpenAI calls per analysis (default: 3)
- `OPENAI_CALL_DELAY`: Delay between OpenAI API calls (default: 2 seconds)

### Customization Options
- Adjust debate rounds (currently set to maximum 5)
- Modify agent prompts for different analysis styles
- Configure risk tolerance parameters
- Customize confidence thresholds

## Deployment

The system is deployment-ready for various platforms:

### Streamlit Cloud
1. Push code to GitHub repository
2. Connect repository to Streamlit Cloud
3. Add environment variables in Streamlit dashboard
4. Deploy with one click

### Docker Deployment
```bash
docker build -t alphaagents .
docker run -p 8501:8501 --env-file .env alphaagents
```

### Cloud Platforms
- **Heroku**: Use provided Procfile
- **Railway**: Auto-detects Python application
- **Render**: Configure build and start commands

## Contributing

Contributions are welcome! Areas for enhancement include:
- Additional agent types (Technical Analysis, ESG, Macroeconomic)
- Advanced debate mechanisms and consensus algorithms
- Portfolio-level analysis capabilities
- Historical backtesting and performance metrics
- Enhanced visualization and reporting features

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This system is inspired by research in multi-agent systems for financial decision-making, implementing collaborative AI approaches to reduce bias and improve accuracy in equity analysis.

## Support

For questions, issues, or feature requests, please open an issue on the GitHub repository or contact the development team.

---

**Built for intelligent investment analysis through collaborative AI**