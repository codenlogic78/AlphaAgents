#!/usr/bin/env python3
"""
Document analysis for stock research
"""

import os
import requests
from typing import List, Dict, Any
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

class FundamentalRAG:
    """Analyzes company documents"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def get_fundamental_rag_analysis(self, symbol: str) -> Dict:
        """Analyze a stock using documents"""
        try:
            # Get company info from different sources
            documents = self._gather_company_documents(symbol)
            if not documents:
                return {"error": f"Couldn't find info for {symbol}"}
            
            # Break documents into searchable pieces
            chunks = self._create_document_chunks(documents)
            
            # Create search index
            vectors = self._create_vectors(chunks)
            
            # Analyze key business areas
            analysis = self._perform_10k_style_analysis(symbol, chunks, vectors)
            
            return {
                "symbol": symbol,
                "documents_analyzed": len(documents),
                "analysis": analysis,
                "data_source": "FUNDAMENTAL_RAG",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Fundamental RAG failed for {symbol}: {str(e)}"}
    
    def _gather_company_documents(self, symbol: str) -> List[Dict]:
        """Collect company info from various sources"""
        documents = []
        
        # Get basic company info and financials
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Add business description if available
            if info.get('longBusinessSummary'):
                documents.append({
                    'type': 'business_overview',
                    'content': info['longBusinessSummary'],
                    'source': 'company_profile'
                })
            
            # Create a financial summary
            financial_summary = self._create_financial_summary(info)
            if financial_summary:
                documents.append({
                    'type': 'financial_data',
                    'content': financial_summary,
                    'source': 'financial_metrics'
                })
                
        except Exception as e:
            print(f"Couldn't get basic info: {e}")
        
        # Try to get company website data
        try:
            company_data = self._get_company_website_data(symbol)
            if company_data:
                documents.extend(company_data)
        except:
            pass
        
        # Get recent news for context
        try:
            news_data = self._get_financial_news_documents(symbol)
            if news_data:
                documents.extend(news_data)
        except:
            pass
            
        return documents
    
    def _create_financial_summary(self, info: Dict) -> str:
        """Turn raw financial data into readable summary"""
        try:
            summary_parts = []
            
            # Basic money stuff
            if info.get('totalRevenue'):
                summary_parts.append(f"Revenue: ${info['totalRevenue']:,}")
            if info.get('grossMargins'):
                summary_parts.append(f"Gross Margin: {info['grossMargins']:.2%}")
            if info.get('profitMargins'):
                summary_parts.append(f"Profit Margin: {info['profitMargins']:.2%}")
            
            # Financial health
            if info.get('totalCash'):
                summary_parts.append(f"Cash: ${info['totalCash']:,}")
            if info.get('totalDebt'):
                summary_parts.append(f"Debt: ${info['totalDebt']:,}")
            if info.get('returnOnEquity'):
                summary_parts.append(f"ROE: {info['returnOnEquity']:.2%}")
            
            # Size metrics
            if info.get('marketCap'):
                summary_parts.append(f"Market Cap: ${info['marketCap']:,}")
            if info.get('enterpriseValue'):
                summary_parts.append(f"Enterprise Value: ${info['enterpriseValue']:,}")
            
            # Growth numbers
            if info.get('revenueGrowth'):
                summary_parts.append(f"Revenue Growth: {info['revenueGrowth']:.2%}")
            if info.get('earningsGrowth'):
                summary_parts.append(f"Earnings Growth: {info['earningsGrowth']:.2%}")
            
            # Things to watch out for
            risk_factors = []
            if info.get('debtToEquity', 0) > 100:
                risk_factors.append("High debt levels could be risky")
            if info.get('beta', 1) > 1.5:
                risk_factors.append("Stock is more volatile than market")
            if info.get('currentRatio', 1) < 1:
                risk_factors.append("Might have trouble paying short-term bills")
            
            if risk_factors:
                summary_parts.append("Risks: " + "; ".join(risk_factors))
            
            return ". ".join(summary_parts) + "."
            
        except Exception:
            return ""
    
    def _get_company_website_data(self, symbol: str) -> List[Dict]:
        """Could scrape company websites but skipping for now"""
        # Maybe add this later - would need to handle different site structures
        return []
    
    def _get_financial_news_documents(self, symbol: str) -> List[Dict]:
        """Grab recent news to understand what's happening"""
        documents = []
        
        try:
            # Get recent news from yfinance
            stock = yf.Ticker(symbol)
            news = stock.news
            
            for article in news[:5]:  # Just grab the top 5
                if article.get('title') and article.get('summary'):
                    content = f"{article['title']}. {article.get('summary', '')}"
                    documents.append({
                        'type': 'recent_developments',
                        'content': content,
                        'source': 'financial_news'
                    })
                    
        except Exception:
            pass
            
        return documents
    
    def _create_document_chunks(self, documents: List[Dict]) -> List[str]:
        """Break up documents into smaller searchable pieces"""
        chunks = []
        
        for doc in documents:
            content = doc.get('content', '')
            doc_type = doc.get('type', 'general')
            
            # Split up long documents
            if len(content) > 500:
                # Break by sentences so we don't cut off mid-thought
                sentences = content.split('.')
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) < 400:
                        current_chunk += sentence + "."
                    else:
                        if current_chunk.strip():
                            chunks.append(f"[{doc_type.upper()}] {current_chunk.strip()}")
                        current_chunk = sentence + "."
                
                if current_chunk.strip():
                    chunks.append(f"[{doc_type.upper()}] {current_chunk.strip()}")
            else:
                if content.strip():
                    chunks.append(f"[{doc_type.upper()}] {content.strip()}")
        
        return chunks
    
    def _create_vectors(self, chunks: List[str]) -> np.ndarray:
        """Turn text into numbers for similarity search"""
        if not chunks:
            return np.array([])
        
        try:
            vectors = self.vectorizer.fit_transform(chunks)
            return vectors.toarray()
        except:
            return np.array([])
    
    def _perform_10k_style_analysis(self, symbol: str, chunks: List[str], vectors: np.ndarray) -> Dict:
        """Look at the company from different angles like a real analyst would"""
        if len(chunks) == 0 or vectors.size == 0:
            return {"error": "No content to analyze"}
        
        # Different areas to investigate
        analysis_queries = {
            'business_model': 'business operations products services revenue model competitive advantages',
            'financial_performance': 'revenue profit earnings cash flow financial performance results',
            'risk_factors': 'risks challenges uncertainties threats competitive risks market risks',
            'growth_prospects': 'growth opportunities expansion plans future outlook development',
            'competitive_position': 'competition market position competitive advantages industry leadership'
        }
        
        analysis_results = {}
        
        # Search for info on each topic
        for section, query in analysis_queries.items():
            relevant_chunks = self._query_chunks(query, chunks, vectors)
            insights = self._extract_section_insights(relevant_chunks, section)
            analysis_results[section] = insights
        
        # Put it all together for a final recommendation
        recommendation = self._generate_fundamental_recommendation(analysis_results)
        
        return {
            "recommendation": recommendation["action"],
            "confidence": recommendation["confidence"],
            "reasoning": recommendation["reasoning"],
            "sections": analysis_results,
            "total_chunks_analyzed": len(chunks)
        }
    
    def _query_chunks(self, query: str, chunks: List[str], vectors: np.ndarray, top_k: int = 3) -> List[str]:
        """Find the most relevant text pieces for what we're looking for"""
        try:
            # Turn the search query into numbers
            query_vector = self.vectorizer.transform([query]).toarray()
            
            # See how similar each chunk is to what we want
            similarities = cosine_similarity(query_vector, vectors)[0]
            
            # Get the best matches
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            return [chunks[i] for i in top_indices if similarities[i] > 0.05]
        except:
            return chunks[:top_k]  # Just take the first few if something breaks
    
    def _extract_section_insights(self, chunks: List[str], section: str) -> Dict:
        """Pull out the important stuff from each business area"""
        if not chunks:
            return {"score": 0, "key_points": [], "sentiment": "neutral"}
        
        # Mash all the relevant text together
        combined_text = ' '.join(chunks).lower()
        
        # Score different aspects
        if section == 'business_model':
            score = self._score_business_strength(combined_text)
        elif section == 'financial_performance':
            score = self._score_financial_performance(combined_text)
        elif section == 'risk_factors':
            score = self._score_risk_level(combined_text)  # Negative = bad
        elif section == 'growth_prospects':
            score = self._score_growth_potential(combined_text)
        elif section == 'competitive_position':
            score = self._score_competitive_strength(combined_text)
        else:
            score = 0
        
        # Grab some key quotes
        key_points = []
        for chunk in chunks[:2]:
            # Clean up the text labels
            clean_chunk = chunk.replace('[BUSINESS_OVERVIEW]', '').replace('[FINANCIAL_DATA]', '').replace('[RECENT_DEVELOPMENTS]', '').strip()
            sentences = clean_chunk.split('.')
            for sentence in sentences:
                if 20 < len(sentence.strip()) < 120:  # Not too short or long
                    key_points.append(sentence.strip())
                    if len(key_points) >= 2:
                        break
        
        return {
            "score": score,
            "key_points": key_points[:2],
            "sentiment": "positive" if score > 2 else "negative" if score < -2 else "neutral"
        }
    
    def _score_business_strength(self, text: str) -> int:
        """How strong is their business model?"""
        positive = ['leading', 'dominant', 'innovative', 'competitive advantage', 'market leader', 'strong brand', 'diversified']
        negative = ['declining', 'competitive pressure', 'market share loss', 'outdated', 'struggling']
        
        pos_count = sum(1 for word in positive if word in text)
        neg_count = sum(1 for word in negative if word in text)
        
        return min(pos_count - neg_count, 5)
    
    def _score_financial_performance(self, text: str) -> int:
        """Are they making money?"""
        positive = ['revenue growth', 'profit increase', 'strong earnings', 'cash generation', 'margin expansion']
        negative = ['revenue decline', 'loss', 'margin compression', 'cash burn', 'debt increase']
        
        pos_count = sum(1 for phrase in positive if phrase in text)
        neg_count = sum(1 for phrase in negative if phrase in text)
        
        return min(pos_count - neg_count, 5)
    
    def _score_risk_level(self, text: str) -> int:
        """What could go wrong? (negative = more risky)"""
        risk_indicators = ['high debt', 'regulatory risk', 'competition', 'market volatility', 'liquidity risk']
        risk_count = sum(1 for risk in risk_indicators if risk in text)
        
        return -min(risk_count, 5)  # Negative because risks are bad
    
    def _score_growth_potential(self, text: str) -> int:
        """Can they grow from here?"""
        growth_indicators = ['expansion', 'new markets', 'innovation', 'development', 'growth opportunities']
        decline_indicators = ['mature market', 'declining demand', 'saturation']
        
        growth_count = sum(1 for indicator in growth_indicators if indicator in text)
        decline_count = sum(1 for indicator in decline_indicators if indicator in text)
        
        return min(growth_count - decline_count, 5)
    
    def _score_competitive_strength(self, text: str) -> int:
        """How do they stack up vs competitors?"""
        strength_indicators = ['market leader', 'competitive advantage', 'moat', 'differentiation', 'brand strength']
        weakness_indicators = ['intense competition', 'commoditized', 'price pressure']
        
        strength_count = sum(1 for indicator in strength_indicators if indicator in text)
        weakness_count = sum(1 for indicator in weakness_indicators if indicator in text)
        
        return min(strength_count - weakness_count, 5)
    
    def _generate_fundamental_recommendation(self, analysis: Dict) -> Dict:
        """Put all the pieces together for a final call"""
        # Get scores from each area
        business_score = analysis.get('business_model', {}).get('score', 0)
        financial_score = analysis.get('financial_performance', {}).get('score', 0)
        risk_score = analysis.get('risk_factors', {}).get('score', 0)  # Already negative
        growth_score = analysis.get('growth_prospects', {}).get('score', 0)
        competitive_score = analysis.get('competitive_position', {}).get('score', 0)
        
        # Weight them by importance (financials matter most)
        total_score = (business_score * 0.25 + 
                      financial_score * 0.30 + 
                      risk_score * 0.15 +  # Risk score is negative
                      growth_score * 0.20 + 
                      competitive_score * 0.10)
        
        # Make the call
        if total_score >= 3:
            action = "BUY"
            confidence = "HIGH"
            reasoning = f"Looks really strong across the board (score: {total_score:.1f})"
        elif total_score >= 1:
            action = "BUY"
            confidence = "MEDIUM"
            reasoning = f"Generally positive fundamentals (score: {total_score:.1f})"
        elif total_score <= -3:
            action = "SELL"
            confidence = "HIGH"
            reasoning = f"Multiple red flags in the analysis (score: {total_score:.1f})"
        elif total_score <= -1:
            action = "SELL"
            confidence = "MEDIUM"
            reasoning = f"Some concerning signs (score: {total_score:.1f})"
        else:
            action = "HOLD"
            confidence = "MEDIUM"
            reasoning = f"Mixed signals - nothing clear either way (score: {total_score:.1f})"
        
        return {
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
            "total_score": round(total_score, 2)
        }

# Create the analyzer
fundamental_rag = FundamentalRAG()

def get_fundamental_rag_data(symbol: str) -> str:
    """Main function that other parts of the system call"""
    data = fundamental_rag.get_fundamental_rag_analysis(symbol)
    return str(data)
