import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Configuration from settings - Increased delays for real API calls
API_DELAY = int(os.getenv("API_DELAY", 3))  # Increased from 1 to 3 seconds
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
CACHE_DURATION = int(os.getenv("CACHE_DURATION", 3600))

class YFinanceTools:
    """Financial data collection tools using yfinance (AlphaAgents Paper Implementation)"""
    
    def __init__(self):
        self.cache = {}
        
    def get_stock_fundamentals(self, symbol: str) -> dict:
        """Get fundamental data for Fundamental Agent (Paper's 10K report focus)"""
        try:
            time.sleep(API_DELAY)
            
            # New yfinance version handles sessions automatically
            stock = yf.Ticker(symbol)
            
            # Try multiple methods to get data
            info = None
            try:
                info = stock.info
            except:
                # Fallback method
                stock = yf.Ticker(symbol)
                info = stock.fast_info.__dict__ if hasattr(stock, 'fast_info') else {}
            
            if not info or len(info) < 3:
                # Try alternative ticker format
                alt_symbol = symbol.replace('.', '-') if '.' in symbol else symbol
                if alt_symbol != symbol:
                    stock = yf.Ticker(alt_symbol, session=session)
                    info = stock.info
                
                if not info or len(info) < 3:
                    return {"error": f"No fundamental data available for {symbol}"}
            
            # Extract available data
            fundamentals = {
                "symbol": symbol,
                "market_cap": info.get('marketCap') or info.get('market_cap', 'N/A'),
                "pe_ratio": info.get('trailingPE') or info.get('pe_ratio', 'N/A'),
                "forward_pe": info.get('forwardPE') or info.get('forward_pe', 'N/A'),
                "price_to_book": info.get('priceToBook') or info.get('pb_ratio', 'N/A'),
                "debt_to_equity": info.get('debtToEquity') or info.get('debt_equity', 'N/A'),
                "roe": info.get('returnOnEquity') or info.get('roe', 'N/A'),
                "profit_margin": info.get('profitMargins') or info.get('profit_margin', 'N/A'),
                "revenue_growth": info.get('revenueGrowth') or info.get('revenue_growth', 'N/A'),
                "earnings_growth": info.get('earningsGrowth') or info.get('earnings_growth', 'N/A'),
                "current_ratio": info.get('currentRatio') or info.get('current_ratio', 'N/A'),
                "recommendation": info.get('recommendationKey', 'hold'),
                "target_price": info.get('targetMeanPrice') or info.get('target_price', 'N/A'),
                "analyst_count": info.get('numberOfAnalystOpinions', 0),
                "current_price": info.get('currentPrice') or info.get('regularMarketPrice', 'N/A'),
                "financial_health": self._assess_financial_health(info),
                "data_source": "yfinance_api"
            }
            
            return fundamentals
            
        except Exception as e:
            return {"error": f"Failed to get fundamentals for {symbol}: {str(e)}"}
    
    def get_stock_valuation_trends(self, symbol: str, period: str = "1y") -> dict:
        """Get valuation trends for Valuation Agent (Paper's extended time horizon focus)"""
        try:
            time.sleep(API_DELAY)
            
            # New yfinance version handles sessions automatically
            stock = yf.Ticker(symbol)
            
            # Try different periods and methods
            periods_to_try = [period, "6mo", "3mo", "1mo", "5d"]
            hist = None
            
            for p in periods_to_try:
                try:
                    hist = stock.history(period=p, interval="1d")
                    if not hist.empty:
                        break
                except:
                    continue
            
            # If still no data, try different approach
            if hist is None or hist.empty:
                try:
                    # Try getting recent data with different method
                    hist = stock.history(start="2024-01-01")
                    if hist.empty:
                        hist = stock.history(period="1mo", interval="1d")
                except:
                    pass
            
            if hist is None or hist.empty:
                return {"error": f"No historical price data available for {symbol} from yfinance API"}
            
            # Paper focuses on "valuation trends over extended time horizon"
            
            if hist.empty:
                return {"error": f"No price data found for {symbol}"}
            
            # Add technical indicators for valuation analysis
            hist = self._add_technical_indicators(hist)
            
            valuation_data = {
                "symbol": symbol,
                "historical_data": hist.to_dict(),
                "current_price": float(hist['Close'].iloc[-1]),
                "price_trends": self._analyze_price_trends(hist),
                "volume_analysis": self._analyze_volume_patterns(hist),
                "volatility_metrics": self._calculate_volatility_metrics(hist),
                "valuation_metrics": {
                    "market_cap": stock.info.get('marketCap', 0) if stock.info else 0,
                    "enterprise_value": stock.info.get('enterpriseValue', 0) if stock.info else 0,
                    "pe_ratio": stock.info.get('trailingPE', None) if stock.info else None,
                    "pb_ratio": stock.info.get('priceToBook', None) if stock.info else None,
                    "ps_ratio": stock.info.get('priceToSalesTrailing12Months', None) if stock.info else None
                },
                "technical_indicators": self._get_current_technical_signals(hist)
            }
            
            return valuation_data
            
        except Exception as e:
            return {"error": f"Failed to get valuation data for {symbol}: {str(e)}"}
    
    def _assess_financial_health(self, info: dict) -> str:
        """Assess overall financial health for real analysis"""
        try:
            score = 0
            
            # P/E ratio assessment
            pe = info.get('trailingPE')
            if pe and 10 <= pe <= 25:
                score += 1
            elif pe and pe < 10:
                score += 0.5
            
            # Debt to equity assessment
            debt_equity = info.get('debtToEquity')
            if debt_equity and debt_equity < 50:
                score += 1
            elif debt_equity and debt_equity < 100:
                score += 0.5
            
            # ROE assessment
            roe = info.get('returnOnEquity')
            if roe and roe > 0.15:
                score += 1
            elif roe and roe > 0.10:
                score += 0.5
            
            # Profit margin assessment
            profit_margin = info.get('profitMargins')
            if profit_margin and profit_margin > 0.15:
                score += 1
            elif profit_margin and profit_margin > 0.05:
                score += 0.5
            
            # Revenue growth assessment
            revenue_growth = info.get('revenueGrowth')
            if revenue_growth and revenue_growth > 0.10:
                score += 1
            elif revenue_growth and revenue_growth > 0.05:
                score += 0.5
            
            # Convert score to health assessment
            if score >= 4:
                return "excellent"
            elif score >= 3:
                return "good"
            elif score >= 2:
                return "fair"
            elif score >= 1:
                return "poor"
            else:
                return "weak"
                
        except:
            return "unknown"
    
    def _calculate_financial_ratios(self, stock) -> dict:
        """Calculate key financial ratios for fundamental analysis"""
        try:
            info = stock.info
            if not info:
                return {}
                
            return {
                "profitability": {
                    "roe": info.get('returnOnEquity', None),
                    "roa": info.get('returnOnAssets', None),
                    "profit_margin": info.get('profitMargins', None),
                    "operating_margin": info.get('operatingMargins', None)
                },
                "liquidity": {
                    "current_ratio": info.get('currentRatio', None),
                    "quick_ratio": info.get('quickRatio', None)
                },
                "leverage": {
                    "debt_to_equity": info.get('debtToEquity', None),
                    "total_debt": info.get('totalDebt', None)
                },
                "efficiency": {
                    "asset_turnover": info.get('totalRevenue', 0) / info.get('totalAssets', 1) if info.get('totalAssets') else None,
                    "inventory_turnover": info.get('totalRevenue', 0) / info.get('inventory', 1) if info.get('inventory') else None
                }
            }
        except:
            return {}
    
    def _analyze_price_trends(self, df: pd.DataFrame) -> dict:
        """Analyze price trends for valuation agent"""
        try:
            current_price = df['Close'].iloc[-1]
            price_30d = df['Close'].iloc[-30] if len(df) >= 30 else df['Close'].iloc[0]
            price_90d = df['Close'].iloc[-90] if len(df) >= 90 else df['Close'].iloc[0]
            price_1y = df['Close'].iloc[0]
            
            return {
                "trend_30d": (current_price - price_30d) / price_30d * 100,
                "trend_90d": (current_price - price_90d) / price_90d * 100,
                "trend_1y": (current_price - price_1y) / price_1y * 100,
                "price_momentum": "bullish" if current_price > df['SMA_20'].iloc[-1] else "bearish"
            }
        except:
            return {}
    
    def _analyze_volume_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze volume patterns for liquidity assessment"""
        try:
            avg_volume = df['Volume'].mean()
            recent_volume = df['Volume'].iloc[-5:].mean()  # Last 5 days
            
            return {
                "average_volume": int(avg_volume),
                "recent_volume": int(recent_volume),
                "volume_trend": "increasing" if recent_volume > avg_volume * 1.2 else "decreasing" if recent_volume < avg_volume * 0.8 else "stable",
                "liquidity_score": "high" if avg_volume > 1000000 else "medium" if avg_volume > 100000 else "low"
            }
        except:
            return {}
    
    def _calculate_volatility_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate volatility metrics for risk assessment"""
        try:
            returns = df['Close'].pct_change().dropna()
            
            return {
                "daily_volatility": float(returns.std()),
                "annualized_volatility": float(returns.std() * np.sqrt(252)),
                "max_drawdown": float((df['Close'] / df['Close'].cummax() - 1).min()),
                "volatility_regime": "high" if returns.std() > 0.03 else "medium" if returns.std() > 0.015 else "low"
            }
        except:
            return {}
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for valuation analysis"""
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # MACD
        df = self._calculate_macd(df)
        
        # Bollinger Bands
        df = self._calculate_bollinger_bands(df)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator"""
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        sma = df['Close'].rolling(window=window).mean()
        std = df['Close'].rolling(window=window).std()
        df['BB_Upper'] = sma + (std * 2)
        df['BB_Lower'] = sma - (std * 2)
        df['BB_Middle'] = sma
        return df
    
    def _get_current_technical_signals(self, df: pd.DataFrame) -> dict:
        """Get current technical analysis signals"""
        try:
            latest = df.iloc[-1]
            
            # RSI signals
            rsi_signal = "overbought" if latest['RSI'] > 70 else "oversold" if latest['RSI'] < 30 else "neutral"
            
            # MACD signals
            macd_signal = "bullish" if latest['MACD'] > latest['MACD_Signal'] else "bearish"
            
            # Moving average signals
            ma_signal = "bullish" if latest['Close'] > latest['SMA_20'] > latest['SMA_50'] else "bearish"
            
            # Bollinger Bands signals
            bb_signal = "overbought" if latest['Close'] > latest['BB_Upper'] else "oversold" if latest['Close'] < latest['BB_Lower'] else "neutral"
            
            return {
                "rsi_signal": rsi_signal,
                "macd_signal": macd_signal,
                "moving_average_signal": ma_signal,
                "bollinger_bands_signal": bb_signal,
                "overall_technical_sentiment": self._determine_overall_signal([rsi_signal, macd_signal, ma_signal, bb_signal])
            }
        except:
            return {}
    
    def _determine_overall_signal(self, signals: list) -> str:
        """Determine overall technical signal from individual indicators"""
        bullish_count = signals.count("bullish")
        bearish_count = signals.count("bearish")
        
        if bullish_count > bearish_count:
            return "bullish"
        elif bearish_count > bullish_count:
            return "bearish"
        else:
            return "neutral"

# Create global instance
yfinance_tools = YFinanceTools()

# Export functions for AutoGen agents (Paper's exact architecture)
def get_fundamental_data(symbol: str) -> str:
    """Function for Fundamental Agent - 10K report analysis focus"""
    data = yfinance_tools.get_stock_fundamentals(symbol)
    return str(data)

def get_sentiment_data(symbol: str) -> str:
    """Function for Sentiment Agent - NewsAPI financial news (Paper-compliant)"""
    try:
        from .news_tools import get_news_sentiment_data
        return get_news_sentiment_data(symbol)
    except ImportError:
        # Fallback to yfinance news if news_tools not available
        return "NewsAPI tools not available. Please create news_tools.py file."

def get_valuation_data(symbol: str) -> str:
    """Function for Valuation Agent - extended time horizon trends focus"""
    data = yfinance_tools.get_stock_valuation_trends(symbol)
    return str(data)

# Backup yfinance sentiment function (if needed)
def get_sentiment_data_yfinance_backup(symbol: str) -> str:
    """Backup: yfinance sentiment data (not paper-compliant)"""
    try:
        time.sleep(API_DELAY)
        stock = yf.Ticker(symbol)
        
        sentiment_data = {
            "symbol": symbol,
            "news": stock.news[:10] if stock.news else [],
            "recommendations": stock.recommendations.to_dict() if stock.recommendations is not None else {},
            "data_source": "yfinance (backup - not paper-compliant)"
        }
        
        return str(sentiment_data)
    except Exception as e:
        return str({"error": f"Failed to get yfinance sentiment for {symbol}: {str(e)}"})