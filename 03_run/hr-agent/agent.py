import os
from dotenv import load_dotenv

from google.adk.agents.llm_agent import LlmAgent
from google.genai import types
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag

from google.cloud import modelarmor_v1
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

from typing import Optional


#-------------------
# settings
#-------------------
load_dotenv()
RAG_CORPUS = os.getenv("RAG_CORPUS")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

MODELARMOR_LOCATION = os.getenv("MODELARMOR_LOCATION", "us")
TEMPLATE_NAME       = os.getenv("TEMPLATE_NAME")

#model="gemini-3.1-flash-lite-preview"
model="gemini-3.1-pro-preview"


#-----------------
# tools
#-----------------
class ModelArmorGuard:
    def __init__(self, project_id: str, location: str, template_name: str):
        self.template_name = template_name
        self.client = modelarmor_v1.ModelArmorClient(
            client_options={
                "api_endpoint": f"modelarmor.{location}.rep.googleapis.com"
            }
        )

    def before_model_callback(self, callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
        """Sanitize user input before Gemini + RagEngine"""
        user_text = ""
        if llm_request.contents:
            for part in reversed(llm_request.contents):
                if part.role == "user" and part.parts:
                    user_text = part.parts[0].text or ""
                    break

        if not user_text:
            return None

        try:
            response = self.client.sanitize_user_prompt(
                modelarmor_v1.SanitizeUserPromptRequest(
                    name=self.template_name,
                    user_prompt_data=modelarmor_v1.DataItem(text=user_text),
                )
            )
            result = response.sanitization_result
            # uncomment to see JSON response that gets returned
            #print(f"Sanitized result: {result}")

            if result.filter_match_state == modelarmor_v1.FilterMatchState.MATCH_FOUND:
                blocked_by = get_matched_filters(result.filter_results)
                #print(f"Input blocked by Model Armor: {blocked_by}")
                return LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text=(
                            "Request was flagged by our security filters."
                        ))],
                    )
                )
        except Exception as e:
            print(f"Callback error: {e}")

        return None


def get_matched_filters(filter_results: dict) -> list[str]:
    """Find matched filters"""
    matched = []
    MATCH_FOUND = modelarmor_v1.FilterMatchState.MATCH_FOUND

    for key, value in filter_results.items():
        try:
            if key == "rai":    # responsible AI
                if value.rai_filter_result.match_state == MATCH_FOUND:
                    matched.append("rai")
            elif key == "pi_and_jailbreak":
                if value.pi_and_jailbreak_filter_result.match_state == MATCH_FOUND:
                    matched.append("pi_and_jailbreak")
            elif key == "sdp":  # sensitive data protection
                if value.sdp_filter_result.match_state == MATCH_FOUND:
                    matched.append("sdp")
            elif key == "malicious_uris":
                if value.malicious_uris_filter_result.match_state == MATCH_FOUND:
                    matched.append("sdp")
            elif key == "csam":  # child sexual abuse material
                if value.csam_filter_result.match_state == MATCH_FOUND:
                    matched.append("csam")
        except AttributeError:
            matched.append(f"{key} (unknown structure)")

    return matched


query_hr = VertexAiRagRetrieval(
    name="query_rag_engine_corpus",
    description="Use this tool to reieve documentation and reference materiels for the question from the RAG corpus",
    rag_resources=[
        rag.RagResource(
            # RAG corpus resource name
            rag_corpus=RAG_CORPUS
        )
    ],
    similarity_top_k=10,
    vector_distance_threshold=0.6,
)


#-----------------
# agents
#-----------------
guard = ModelArmorGuard(
    project_id=PROJECT_ID,
    location=MODELARMOR_LOCATION,
    template_name=TEMPLATE_NAME,
)

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
        "2. You MUST cite your sources using the format: (Source: [Source Name]).\n"
        "3. If the tool results do not contain the answer, state: 'I'm sorry, I couldn't find that in our HR documents.'\n"
        "4. Do not use outside knowledge or make up facts about company policy."
    ),
    before_model_callback=guard.before_model_callback,
    tools=[query_hr],
)

root_agent = hr_agent
