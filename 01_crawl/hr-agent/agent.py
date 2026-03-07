from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools import FunctionTool
from google.genai import types

from .tools import query_hr

hr_rag_tool = FunctionTool(func=query_hr)

#-------------------
# settings
#-------------------
#model="gemini-3.1-flash-lite-preview"
model="gemini-3.1-pro-preview"


#-----------------
# agents
#-----------------
hr_agent = LlmAgent(
    name="hr_agent",
    model=model,
    description="Specialist in company HR policies and procedures.",
    instruction=(
        "You are a professional HR assistant. Your goal is to answer questions "
        "using ONLY the information retrieved from the 'query_hr' tool. "
        "When calling the 'query_hr' tool, ensure all string arguments are properly formatted as standard JSON strings with double quotes.\n\n"
        "RULES:\n"
        "1. If the tool returns relevant information, summarize it clearly.\n"
        "2. You MUST cite your sources using the format: (Source: [Source Name], Page: [Page Number]).\n"
        "3. If the tool results do not contain the answer, state: 'I'm sorry, I couldn't find that in our HR documents.'\n"
        "4. Do not use outside knowledge or make up facts about company policy."
    ),
    tools=[query_hr],
)

root_agent = hr_agent
