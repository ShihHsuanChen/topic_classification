from typing import List, Dict, Optional, Any, Union

import requests
from pydantic import BaseModel, Field, SecretStr, field_serializer


class GeminiRequestQuery(BaseModel):
    key: SecretStr

    @field_serializer('key', when_used='always')
    def dump_secret(self, v):
        return v.get_secret_value()


class GeminiContent(BaseModel):
    parts: List[Dict[str, str]] = []

    def get_text(self) -> str: # TODO: more cases
        for part in self.parts:
            if 'text' in part:
                return part['text']
        return ''


class GeminiContentWithRole(GeminiContent):
    role: Optional[str] = None


class GeminiErrorContent(BaseModel):
    code: int
    message: str
    status: str


class GeminiResponseError(BaseModel):
    error: GeminiErrorContent


class GeminiRequestData(BaseModel):
    contents: List[GeminiContent] = Field(min_length=1)

    @classmethod
    def from_prompt(cls, prompt: str):
        return cls(contents=[GeminiContent(parts=[{"text": prompt}])])


class GeminiCandidates(BaseModel):
    content: GeminiContentWithRole
    finishReason: Optional[str] = None
    avgLogProbs: Optional[float] = None


class GeminiResponseData(BaseModel):
    candidates: List[GeminiCandidates]
    usageMetadata: Dict[str, Any]
    modelVersion: str
    responseId: str

    def get_text(self):
        if not self.candidates:
            return ''
        return self.candidates[0].content.get_text()


class GeminiAPI:
    def __init__(self, api_key: SecretStr):
        self.api_key = api_key

    def __call__(self, prompt: str) -> Union[GeminiResponseData, GeminiResponseError]:
        resp = requests.post(
            'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent',
            json=GeminiRequestData.from_prompt(prompt).model_dump(),
            params=GeminiRequestQuery(key=self.api_key).model_dump(),
            headers={'Content-Type': 'application/json'},
        )
        if resp.status_code != 200:
            return GeminiResponseError(**resp.json())
        else:
            return GeminiResponseData(**resp.json())


