import abc
import time
from typing import List

from pydantic import SecretStr

from .gemini import GeminiAPI, GeminiResponseData, GeminiResponseError
from .exceptions import MaximumRetryError


class LLMError(ValueError):
    ...


class BaseLLM(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt, **kwargs) -> str:
        raise NotImplementedError()

    def __call__(self, prompt, **kwargs) -> str:
        return self.generate(prompt, **kwargs)


class BaseAPILLM(BaseLLM):
    def __init__(self, retry: int = 3, retry_delay: int = 1, **kwargs):
        self.retry = retry
        self.retry_delay = retry_delay 

    def __call__(self, prompt, **kwargs) -> str:
        excs = []
        for i in range(self.retry + 1):
            if i > 0:
                print(f'Retry-{i}')
            try:
                return self.generate(prompt, **kwargs)
            except Exception as e: # TODO:  more specific error checking
                if self.retry == 0:
                    raise e
                else:
                    excs.append(e)
            time.sleep(self.retry_delay)
        raise MaximumRetryError(excs=excs)


class GeminiAPILLM(BaseAPILLM):
    def __init__(self, api_key: SecretStr, **kwargs):
        super().__init__(**kwargs)
        self.api = GeminiAPI(api_key)

    def generate(self, prompt, **kwargs) -> str:
        res = self.api(prompt, **kwargs)
        if isinstance(res, GeminiResponseData):
            return res.get_text()
        else: # GeminiResponseError
            raise LLMError(res.error.message)
