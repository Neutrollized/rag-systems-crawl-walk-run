import os
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.genai import types

import litellm
litellm.suppress_debug_info = True
litellm.verbose = False

import logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

from .tools import query_hr

hr_rag_tool = FunctionTool(func=query_hr)

#-------------------
# settings
#-------------------
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
MODEL=LiteLlm(model="ollama_chat/gemma4:e4b", extra_body={"think": False})


#-----------------
# agents
#-----------------
hr_agent = LlmAgent(
    name="hr_agent",
    model=MODEL,
    description="Specialist in company HR policies and procedures.",
    instruction=(
        "You are a professional HR assistant. Your goal is to answer questions "
        "using ONLY the information retrieved from the 'hr_rag_tool' tool. "
        "When calling the 'hr_rag_tool' tool, ensure all string arguments are properly formatted as standard JSON strings with double quotes.\n\n"
        "RULES:\n"
        "1. If the tool returns relevant information, summarize it clearly.\n"
        "2. You MUST cite your sources using the format: (Source: [Source Name], Page: [Page Number]).\n"
        "3. If the tool results do not contain the answer, state: 'I'm sorry, I couldn't find that in our HR documents.'\n"
        "4. Do not use outside knowledge or make up facts about company policy."
    ),
    tools=[hr_rag_tool],
)

root_agent = hr_agent
