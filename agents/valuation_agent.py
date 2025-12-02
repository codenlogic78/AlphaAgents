try:
    from autogen import AssistantAgent
except ImportError:
    from autogen_agentchat.agents import AssistantAgent
from config.llm_config import valuation_agent_config
from config.agent_prompts import VALUATION_AGENT_PROMPT
from tools.financial import get_valuation_data

class ValuationAgent(AssistantAgent):
    """Valuation Agent - Analyzes price trends, technical indicators, market valuation"""
    
    def __init__(self):
        try:
            # Old autogen version
            super().__init__(
                name="valuation_agent",
                system_message=VALUATION_AGENT_PROMPT,
                llm_config=valuation_agent_config,
                function_map={
                    "get_valuation_data": get_valuation_data
                }
            )
        except TypeError:
            # New autogen_agentchat version - requires model_client
            from openai import OpenAI
            import os
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            super().__init__(
                name="valuation_agent",
                model_client=client,
                description="Valuation Agent - Analyzes price trends, technical indicators, market valuation"
            )
        
    def analyze_stock(self, symbol: str) -> str:
        """Core valuation analysis for multi-agent debate"""
        try:
            # Get valuation data using yfinance tool
            valuation_data = get_valuation_data(symbol)
            
            # Focused analysis prompt for debate context
            analysis_prompt = f"""
            Analyze {symbol} valuation and technical trends for investment decision:
            
            Data: {valuation_data}
            
            As a valuation analyst, provide:
            1. Current valuation assessment (overvalued/undervalued/fair)
            2. Key valuation metrics (P/E, P/B ratios vs historical/sector)
            3. Technical indicator signals (RSI, MACD, trend analysis)
            4. Price trend analysis over extended time horizon
            5. Your recommendation: BUY/HOLD/SELL with price targets
            
            Focus on quantitative analysis. Be ready to defend your valuation in group discussion.
            """
            
            response = self.generate_reply([{"role": "user", "content": analysis_prompt}])
            return response
            
        except Exception as e:
            return f"Error in valuation analysis for {symbol}: {str(e)}"
