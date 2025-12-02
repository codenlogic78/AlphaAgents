try:
    from autogen import AssistantAgent
except ImportError:
    from autogen_agentchat.agents import AssistantAgent
from config.llm_config import fundamental_agent_config
from config.agent_prompts import FUNDAMENTAL_AGENT_PROMPT
from tools.financial import get_fundamental_data

class FundamentalAgent(AssistantAgent):
    """Fundamental Agent - Analyzes 10-K reports, financial statements, company fundamentals"""
    
    def __init__(self):
        try:
            # Old autogen version
            super().__init__(
                name="fundamental_agent",
                system_message=FUNDAMENTAL_AGENT_PROMPT,
                llm_config=fundamental_agent_config,
                function_map={
                    "get_fundamental_data": get_fundamental_data
                }
            )
        except TypeError:
            # New autogen_agentchat version - requires model_client
            from openai import OpenAI
            import os
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            super().__init__(
                name="fundamental_agent",
                model_client=client,
                description="Fundamental Agent - Analyzes 10-K reports, financial statements, company fundamentals"
            )
        
    def analyze_stock(self, symbol: str) -> str:
        """Core fundamental analysis for multi-agent debate"""
        try:
            # Get fundamental data using the tool
            fundamental_data = get_fundamental_data(symbol)
            
            # Focused analysis prompt for debate context
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
