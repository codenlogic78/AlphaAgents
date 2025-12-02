import os
from dotenv import load_dotenv

#loading enviroment variables
load_dotenv()

#Base LLM configuration for AutoGen
base_llm_config={
    "config_list":[
        {
        "model":"gpt-4-turbo-preview",
        "api_key":os.getenv("OPENAI_API_KEY"),
        "temperature":0.1,
        "max_tokens":2000,
    }
    ],
    "timeout":120,
    "cache_seed":42 #For Reproductible results
}

#Specilization configurations for each Agent
fundamental_agent_config={
    **base_llm_config,
    "config_list":[
        {
            **base_llm_config["config_list"][0],
            "temperature":0.05,
            "max_tokens":2500
        }
    ]

}
sentiment_agent_config={
    **base_llm_config,
    "config_list":[
        {
            **base_llm_config["config_list"][0],
            "temperature":0.2,
            "max_tokens":1500,
        }
    ]
}

valuation_agent_config={
    **base_llm_config,
    "config_list":[
        {
            **base_llm_config["config_list"][0],
            "temperature":0.1,
            "max_tokens":2000,
        }
    ]
}



#Group chat configurations 
group_chat_config={
    "max_round":15,
    "speaker_selection_method":"round_robin",
    "allow_repeat_speaker": False,
}

__all__=[
    "base_llm_config",
    "fundamental_agent_config",
    "sentiment_agent_config",
    "valuation_agent_config",
    "group_chat_config"
]








