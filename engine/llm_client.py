"""
Gemini LLM Client for Conversational Data Analysis using google-genai SDK.
"""

import json
import os
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from .prompts import SYSTEM_PROMPT, ERROR_CORRECTION_PROMPT


class AnalysisResponse(BaseModel):
    thought: str = Field(description="Internal chain of thought and analytical plan")
    needs_query: bool = Field(description="True if SQL query is required to answer the prompt")
    sql_query: Optional[str] = Field(default=None, description="DuckDB SQL query querying the table `dataset`")
    chart_needed: bool = Field(description="True if a chart should be displayed")
    chart_code: Optional[str] = Field(default=None, description="Plotly python code creating `fig` using px or go")
    explanation: str = Field(description="Narrative markdown explanation with quantitative insights and highlights")
    followup_questions: List[str] = Field(default_factory=list, description="2-3 relevant follow-up questions")


class GeminiDataAnalyst:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name
        self.client = None
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)

    def is_configured(self) -> bool:
        return self.client is not None and bool(self.api_key)

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip() if api_key else None
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def analyze(
        self,
        user_query: str,
        schema_context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Send user query and schema context to Gemini to generate analysis plan, SQL query, Plotly code, and explanation.
        """
        if not self.is_configured():
            raise ValueError(
                "Gemini API key is not configured. Please provide your GEMINI_API_KEY in the sidebar or in .env."
            )

        # Build prompt with conversation history context
        history_context = ""
        if chat_history and len(chat_history) > 0:
            history_lines = ["RECENT CONVERSATION HISTORY:"]
            for msg in chat_history[-6:]:  # last 3 turns
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                # Truncate content preview to prevent prompt bloating
                preview = content[:250] + ("..." if len(content) > 250 else "")
                history_lines.append(f"{role}: {preview}")
            history_context = "\n".join(history_lines) + "\n\n"

        prompt = f"""{schema_context}

{history_context}USER QUERY:
{user_query}
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=AnalysisResponse,
                    temperature=0.2,
                ),
            )

            # Parse response
            raw_text = response.text
            data = self._parse_json_response(raw_text)
            return data

        except Exception as e:
            # Fallback to plain text generation if response_schema fails
            try:
                fallback_resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=f"{SYSTEM_PROMPT}\n\nRespond with a valid JSON object matching the required structure.\n\n{prompt}",
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                    ),
                )
                return self._parse_json_response(fallback_resp.text)
            except Exception as e2:
                raise RuntimeError(f"Gemini analysis failed: {str(e2)}")

    def correct_error(
        self,
        user_query: str,
        schema_context: str,
        failed_code: str,
        error_message: str,
    ) -> Dict[str, Any]:
        """Ask Gemini to correct a failed SQL query or chart code."""
        if not self.is_configured():
            raise ValueError("Gemini API key is not configured.")

        error_prompt = ERROR_CORRECTION_PROMPT.format(
            failed_code=failed_code, error_message=error_message
        )
        prompt = f"""{schema_context}

ORIGINAL QUERY: {user_query}

{error_prompt}
"""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=AnalysisResponse,
                temperature=0.1,
            ),
        )
        return self._parse_json_response(response.text)

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Safely parse JSON response from LLM."""
        if not text:
            raise ValueError("Empty response received from LLM.")

        cleaned = text.strip()
        # Remove ```json ... ``` wrapper if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to locate the outermost JSON object
            match = re.search(r"\{[\s\S]*\}", cleaned)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Could not parse valid JSON from LLM response: {text[:200]}")
