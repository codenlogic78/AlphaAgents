import os
from dotenv import load_dotenv
load_dotenv()

#API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEWS_API_KEY=os.getenv("NEWS_API_KEY")

#Project Settings
PROJECT_NAME=os.getenv("PROJECT_NAME", "AlphaAgents")
ENVIRONMENT = os.getenv("ENVIRONMENT","Development")
LOG_LEVEL=os.getenv("LOG_LEVEL","INFO")  

#News Settings
MAX_NEWS_ARTICLES=int(os.getenv("MAX_NEWS_ARTICLES",10))
CACHE_DURATION=int(os.getenv("CACHE_DURATION", 3600))

#Rate Limiting
API_DELAY=int(os.getenv("API_DELAY",1))
MAX_RETRIES=int(os.getenv("MAX_RETRIES",3))

#Data Settings
DEFAULT_STOCK_PERIOD="1y"
TECHNICAL_INDICATORS_PERIOD=14
CACHE_DIRECTORY="data/cache"

#Portfolio Settings
DEFAULT_RISK_TOLERANCE="neutral" #neutral, conservative, aggressive
MAX_PORTFOLIO_SIZE=10
MIN_POSITION_SIZE=0.05

#Logging Configurations
LOGGING_CONFIG={
    "version":1,
    "disable_existing_loggers":False,
    "formatters":{
        "standard":{
            "format":"%(asctime)s[%(levelname)s]%(name)s: %(message)s"
        },
    },
   "handlers":{
    "default":{
        "level":LOG_LEVEL,
        "formatter":"standard",
        "class":"logging.StreamHandler",
    
 },
    "file":{
     "level": LOG_LEVEL,
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "logs/app.log",
            "mode": "a",

 },
},
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": LOG_LEVEL,
            "propagate": False
        }
    }
}

# AutoGen Settings
MAX_ROUNDS = 12
DEFAULT_STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
