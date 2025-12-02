try:
    from autogen import AssistantAgent
except ImportError:
    from autogen_agentchat.agents import AssistantAgent
from config.llm_config import sentiment_agent_config
from config.agent_prompts import SENTIMENT_AGENT_PROMPT
from tools.financial import get_sentiment_data

class SentimentAgent(AssistantAgent):
    """Sentiment Agent - Analyzes financial news, analyst ratings, market sentiment"""
    
    def __init__(self):
        try:
            # Old autogen version
            super().__init__(
                name="sentiment_agent",
                system_message=SENTIMENT_AGENT_PROMPT,
                llm_config=sentiment_agent_config,
                function_map={
                    "get_sentiment_data": get_sentiment_data
                }
            )
        except TypeError:
            # New autogen_agentchat version - requires model_client
            from openai import OpenAI
            import os
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            super().__init__(
                name="sentiment_agent",
                model_client=client,
                description="Sentiment Agent - Analyzes financial news, analyst ratings, market sentiment"
            )
        
    def analyze_stock(self, symbol: str) -> str:
        """Core sentiment analysis for multi-agent debate"""
        try:
            # Get sentiment data using NewsAPI tool
            sentiment_data = get_sentiment_data(symbol)
            
            # Focused analysis prompt for debate context
            analysis_prompt = f"""
            Analyze {symbol} market sentiment for investment decision:
            
            News Data: {sentiment_data}
            
            As a sentiment analyst, provide:
            1. Current market sentiment (positive/negative/neutral) with confidence level
            2. Key news themes driving sentiment
            3. Analyst rating trends (upgrades/downgrades)
            4. Short-term price impact assessment (1-4 weeks)
            5. Your recommendation: BUY/HOLD/SELL based on sentiment
            
            Cite specific news sources and sentiment scores. Be ready to debate with other analysts.
            """
            
            response = self.generate_reply([{"role": "user", "content": analysis_prompt}])
            return response
            
        except Exception as e:
            return f"Error in sentiment analysis for {symbol}: {str(e)}"
