try:
    from autogen import AssistantAgent
except ImportError:
    from autogen_agentchat.agents import AssistantAgent
from config.llm_config import fundamental_agent_config
from config.agent_prompts import FUNDAMENTAL_AGENT_PROMPT
from tools.financial import get_fundamental_data

class FundamentalAgent(AssistantAgent):
    """Fundamental analysis agent"""
    
    def __init__(self):
        try:
            # Old version
            super().__init__(
                name="fundamental_agent",
                system_message=FUNDAMENTAL_AGENT_PROMPT,
                llm_config=fundamental_agent_config,
                function_map={
                    "get_fundamental_data": get_fundamental_data
                }
            )
        except TypeError:
            # New version
            from openai import OpenAI
            import os
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            super().__init__(
                name="fundamental_agent",
                model_client=client,
                description="Fundamental analysis agent"
            )
        
    def analyze_stock(self, symbol: str) -> str:
        """Analyze stock fundamentals"""
        try:
            # Get fundamental data
            fundamental_data = get_fundamental_data(symbol)
            
            # Analysis prompt
            analysis_prompt = f"""
            Analyze {symbol} fundamentals for investment decision:
            
            Data: {fundamental_data}
            
            As a fundamental analyst, provide:
            1. Key financial metrics assessment (revenue, earnings, ratios)
            2. Balance sheet strength evaluation  
            3. Long-term investment thesis
            4. Your recommendation: BUY/HOLD/SELL with reasoning
            
            Focus on facts and data. Be ready to defend your analysis in group discussion.
            """
            
            response = self.generate_reply([{"role": "user", "content": analysis_prompt}])
            return response
            
        except Exception as e:
            return f"Error in fundamental analysis for {symbol}: {str(e)}"
