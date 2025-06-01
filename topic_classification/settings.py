from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config= SettingsConfigDict(
        env_file=('.env'),
        env_file_encoding='utf8',
    )

    gemini_api_key: SecretStr
    batch_size: int = Field(50, ge=1)
