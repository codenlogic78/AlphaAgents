import requests
import os
import time 
from datetime import datetime,timedelta
from dotenv import load_dotenv
load_dotenv()

#Configuration from settings
NEWS_API_KEY=os.getenv("NEWS_API_KEY")
API_DELAY=int(os.getenv("API_DELAY",1))
MAX_NEWS_ARTICLES=int(os.getenv("MAX_NEWS_ARTICLES",10))

class NewsAPITools:
    """NewsAPI tools for Sentiment Agent"""

    def __init__(self):
        self.api_key=NEWS_API_KEY
        self.base_url="https://newsapi.org/v2"

    def get_financial_news_sentiment(self,symbol:str,company_name:str=None)->dict:
        """Get financial news sentiment for Sentiment Agent"""
        try:
            time.sleep(API_DELAY)
            if not self.api_key:
                return{"error":"API key not found"}
            #Search query for the company
            query=f"{symbol}"
            if company_name:
                query+=f" OR \"{company_name}\""

            #NewsAPI parameters - targetting financial news sources
            params={
                'q':query,
                'language':'en',
                'sortBy':'publishedAt',
                'pageSize':MAX_NEWS_ARTICLES,
                'apiKey':self.api_key,
                'domains':'bloomberg.com,reuters.com,cnbc.com,marketwatch.com,finance.yahoo.com,wsj.com,ft.com,barrons.com'

            }

            response=requests.get(f"{self.base_url}/everything",params=params)
            if response.status_code==200:
                data=response.json()
                articles=data.get('articles',[])

                # Process articles for sentiment analysis
                processed_articles=[]
                for article in articles:
                    if article.get('title') and article.get('title')!='[Removed]':
                        processed_article={
                            'title':article.get('title',''),
                            'description':article.get('description',''),
                            'url':article.get('url',''),
                            'publishedAt':article.get('publishedAt',''),
                            'source':article.get('source',{}).get('name',''),
                            'sentiment_analysis': self._analyze_article_sentiment(
                                (article.get('title', '') or '') + ' ' + (article.get('description', '') or '')


                            )
                        }
                        processed_articles.append(processed_article)

                return {
                    "symbol":symbol,
                    "news_articles":processed_articles,
                    "total_articles":len(processed_articles),
                    "overall_sentiment":self._calculate_overall_sentiment(processed_articles),
                    "news_sources":list(set([article['source'] for article in processed_articles if article['source']])),
                    "analyst_ratings_summary":self._extract_rating_mentions(processed_articles),
                    "data_source":"NewsAPI",
                    "timestamp" : datetime.now().isoformat()
                    }
            else:
                return {"error":f"NewsAPI request failed:{response.status_code}-{response.text}"}

        except Exception as e:
            return {"error": f"Failed to get news for {symbol}: {str(e)}"}

    def _analyze_article_sentiment(self,text:str)->dict:
        """Analyze sentiment of individual article"""
        if not text:
            return {"sentiment":"neutral","score":0.0,"keywords":[]}

        text_lower=text.lower()

                #Financial sentiment keywords
        positive_keywords=[
                    'beat','beats','strong','growth',
                    'rise','rises','up','bullish','buy','upgrade','upgrades','outperform',
                    'exceed','exceeds','positive','optimistic','confident','success',
                    'record','high','surge','rally','boost','momentum'

                ]
        negative_keywords=[
            'miss','misses','weak',
            'down','decline','declines','bearish','sell','downgrade','downgrades',
            'underperform','disappoint','disappoints','concern','concerns','worry',
            'risk','risks','low','drop','drops','plunge','crash'
        ]
        found_positive =[word for word in positive_keywords if word in text_lower]
        found_negative=[word for word in negative_keywords if word in text_lower]
        
        #Calculating sentiment count 
        pos_count=len(found_positive)
        neg_count=len(found_negative)
        total_sentiment_words=pos_count+neg_count

        if total_sentiment_words==0:
            sentiment_score=0.0
            sentiment = "neutral"
        else:
            sentiment_score = (pos_count - neg_count) / total_sentiment_words
            
            if sentiment_score > 0.2:
                sentiment = "positive"
            elif sentiment_score < -0.2:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "score": sentiment_score,
            "positive_keywords": found_positive,
            "negative_keywords": found_negative,
            "keyword_count": total_sentiment_words
        }
    
    def _calculate_overall_sentiment(self, articles: list) -> dict:
        """Calculate overall sentiment from all articles"""
        if not articles:
            return {"sentiment": "neutral", "score": 0.0, "confidence": "low"}
        
        total_score = 0
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        total_keywords = 0
        
        for article in articles:
            sentiment_data = article.get('sentiment_analysis', {})
            score = sentiment_data.get('score', 0)
            sentiment = sentiment_data.get('sentiment', 'neutral')
            keywords = sentiment_data.get('keyword_count', 0)
            
            total_score += score
            sentiment_counts[sentiment] += 1
            total_keywords += keywords
        
        # Calculate weighted average
        avg_score = total_score / len(articles)
        
        # Determine overall sentiment
        if avg_score > 0.1:
            overall_sentiment = "positive"
        elif avg_score < -0.1:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        # Determine confidence based on article count and keyword density
        if len(articles) >= 5 and total_keywords >= 10:
            confidence = "high"
        elif len(articles) >= 3 and total_keywords >= 5:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "sentiment": overall_sentiment,
            "score": avg_score,
            "confidence": confidence,
            "article_breakdown": sentiment_counts,
            "total_articles": len(articles),
            "total_sentiment_keywords": total_keywords
        }
    
    def _extract_rating_mentions(self, articles: list) -> dict:
        """Extract analyst rating mentions from news articles"""
        rating_keywords = {
            "upgrades": ["upgrade", "upgrades", "raised", "increase", "buy rating"],
            "downgrades": ["downgrade", "downgrades", "lowered", "decrease", "sell rating"],
            "price_targets": ["price target", "target price", "pt raised", "pt lowered"]
        }
        
        mentions = {"upgrades": 0, "downgrades": 0, "price_targets": 0, "examples": []}
        
        for article in articles:
            text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
            
            for category, keywords in rating_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        mentions[category] += 1
                        mentions["examples"].append({
                            "type": category,
                            "title": article.get('title', ''),
                            "source": article.get('source', '')
                        })
                        break  # Count each article only once per category
        
        return mentions

# Create global instance
news_tools = NewsAPITools()

# Export function for Sentiment Agent (Paper-compliant)
def get_news_sentiment_data(symbol: str, company_name: str = None) -> str:
    """Function for Sentiment Agent - NewsAPI financial news (Paper-compliant Bloomberg equivalent)"""
    data = news_tools.get_financial_news_sentiment(symbol, company_name)
    return str(data)
 

               



