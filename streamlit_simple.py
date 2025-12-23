import streamlit as st
import sys
import os

# Add project path (works both locally and on Streamlit Cloud)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Check for required environment variables
if not os.getenv("OPENAI_API_KEY"):
    st.error("🔑 OpenAI API Key not found! Please add OPENAI_API_KEY to Streamlit Cloud secrets.")
    st.info("Go to your app settings → Secrets → Add: OPENAI_API_KEY = 'your-key-here'")
    st.stop()

# Import your AlphaAgents system
try:
    from main import AlphaAgentsSystem
    MAIN_AVAILABLE = True
except Exception as e:
    st.error(f"Error importing main: {e}")
    MAIN_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="AlphaAgents",
    page_icon="📈",
    layout="wide"
)

# Header
st.title("🤖 AlphaAgents")
st.subheader("Paper-Compliant Multi-Agent Equity Analysis System")

# Sidebar
st.sidebar.header("Analysis Configuration")

# Stock symbol input
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()

# Risk tolerance
risk_tolerance = st.sidebar.selectbox(
    "Risk Tolerance",
    ["conservative", "neutral", "aggressive"],
    index=1
)

# Analysis button
if st.sidebar.button("Analyze Stock", type="primary"):
    if symbol and MAIN_AVAILABLE:
        st.header(f"Analysis for {symbol}")
        
        # Progress
        progress = st.progress(0)
        status = st.empty()
        
        try:
            # Initialize system
            status.text("Initializing AlphaAgents...")
            progress.progress(25)
            
            alphaagents = AlphaAgentsSystem()
            
            # Run analysis
            status.text("Running analysis...")
            progress.progress(50)
            
            result = alphaagents.analyze_stock(symbol, risk_tolerance)
            
            progress.progress(100)
            status.text("Complete!")
            
            # Results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Final Recommendation")
                
                final_rec = result.get('final_recommendation', {})
                recommendation = final_rec.get('recommendation', 'UNKNOWN')
                confidence = final_rec.get('confidence', 'UNKNOWN')
                
                if recommendation == 'BUY':
                    st.success(f"📈 {recommendation}")
                elif recommendation == 'HOLD':
                    st.warning(f"📊 {recommendation}")
                elif recommendation == 'SELL':
                    st.error(f"📉 {recommendation}")
                
                st.write(f"**Confidence:** {confidence}")
                st.write(f"**Method:** {result.get('method', 'unknown')}")
                
                # Show consensus status
                consensus = result.get('consensus_reached', False)
                if consensus:
                    st.success("✅ Consensus Reached")
                else:
                    st.warning("❌ No Consensus - Majority Vote")
            
            with col2:
                st.subheader("🗳️ Vote Summary")
                vote_counts = result.get('vote_counts', {})
                if vote_counts:
                    for vote, count in vote_counts.items():
                        if count > 0:
                            if vote == 'BUY':
                                st.write(f"🟢 {vote}: {count}")
                            elif vote == 'SELL':
                                st.write(f"🔴 {vote}: {count}")
                            else:
                                st.write(f"🟡 {vote}: {count}")
            
            # Debate Section
            st.subheader("🗣️ Agent Debate")
            collab_result = result.get("collaborative_result", {})
            
            if collab_result:
                debate_method = collab_result.get("method", "N/A")
                debate_summary = collab_result.get("debate_summary", "N/A")
                
                if debate_method == "immediate_consensus":
                    st.info(f"✅ **Immediate Consensus:** {debate_summary}")
                elif debate_method == "debate_consensus":
                    rounds = collab_result.get("rounds", "N/A")
                    st.success(f"🎯 **Consensus after {rounds} rounds:** {debate_summary}")
                    
                    # Show debate responses
                    debate_responses = collab_result.get("debate_responses", [])
                    if debate_responses:
                        with st.expander("View Debate Details"):
                            agents = ["Fundamental", "Sentiment", "Valuation"]
                            round_num = 1
                            for i, response in enumerate(debate_responses):
                                agent = agents[i % 3]
                                if i % 3 == 0 and i > 0:
                                    round_num += 1
                                # Clean response text to avoid regex issues
                                clean_response = str(response).replace('$', '\\$').replace('{', '\\{').replace('}', '\\}')
                                st.write(f"**{agent} R{round_num}:** {clean_response}")
                                
                elif debate_method == "majority_voting":
                    st.warning(f"⚖️ **Majority Voting:** {debate_summary}")
                    
                    # Show debate that led to majority vote
                    debate_responses = collab_result.get("debate_responses", [])
                    if debate_responses:
                        with st.expander("View Debate Before Majority Vote"):
                            agents = ["Fundamental", "Sentiment", "Valuation"]
                            round_num = 1
                            for i, response in enumerate(debate_responses):
                                agent = agents[i % 3]
                                if i % 3 == 0 and i > 0:
                                    round_num += 1
                                # Clean response text to avoid regex issues
                                clean_response = str(response).replace('$', '\\$').replace('{', '\\{').replace('}', '\\}')
                                st.write(f"**{agent} R{round_num}:** {clean_response}")
            else:
                st.info("No debate information available")
            
            # Individual agents
            st.subheader("🤖 Individual Agent Analysis")
            
            individual_analyses = result.get('individual_analyses', {})
            if individual_analyses:
                col1, col2, col3 = st.columns(3)
                
                agents = [
                    ('Fundamental', col1, 'fundamental'),
                    ('Sentiment', col2, 'sentiment'), 
                    ('Valuation', col3, 'valuation')
                ]
                
                for agent_name, col, agent_key in agents:
                    if agent_key in individual_analyses:
                        analysis = individual_analyses[agent_key]
                        
                        with col:
                            st.write(f"**{agent_name} Agent**")
                            
                            agent_rec = analysis.get('recommendation', 'UNKNOWN')
                            if agent_rec == 'BUY':
                                st.success(f"📈 {agent_rec}")
                            elif agent_rec == 'HOLD':
                                st.warning(f"📊 {agent_rec}")
                            elif agent_rec == 'SELL':
                                st.error(f"📉 {agent_rec}")
                            else:
                                st.write(f"**{agent_rec}**")
                            
                            confidence = analysis.get('confidence', 'N/A')
                            st.write(f"**Confidence:** {confidence}")
                            
                            reasoning = analysis.get('reasoning', 'No reasoning')
                            # Clean reasoning text to avoid regex issues
                            reasoning = str(reasoning).replace('$', '\\$').replace('{', '\\{').replace('}', '\\}')
                            if len(reasoning) > 100:
                                reasoning = reasoning[:100] + "..."
                            st.write(f"**Reasoning:** {reasoning}")
            
            # Raw data
            with st.expander("Raw Data"):
                st.json(result)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            progress.progress(0)
            status.text("Failed!")
    
    elif not MAIN_AVAILABLE:
        st.error("Main.py import failed. Please check the system.")

else:
    # Welcome screen
    st.header("Welcome to AlphaAgents")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Fundamental Agent")
        st.write("- Financial statements analysis")
        st.write("- Company fundamentals")
        st.write("- Long-term value assessment")
    
    with col2:
        st.subheader("📰 Sentiment Agent")
        st.write("- Financial news analysis")
        st.write("- Market sentiment")
        st.write("- Short-term impact")
    
    with col3:
        st.subheader("📈 Valuation Agent")
        st.write("- Technical analysis")
        st.write("- Price trends")
        st.write("- Market valuation")
    
    st.info("Enter a stock symbol in the sidebar and click 'Analyze Stock' to begin!")
