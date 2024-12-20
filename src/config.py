from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import cache
# from dotenv import load_dotenv

# print(load_dotenv())

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    log_file: str ="detecciones.json"
    api_name: str = "Billete Detector"
    revision: str = "local"
    model_A_path: str = "model/bolivian_money_detector_MK_I.pt"
    llm: str = 'meta-llama/Llama-3.2-1B'
    tts_english:str='tts_models/en/ljspeech/fast_pitch'
    tts_spanish:str='tts_models/es/mai/tacotron2-DDC'
    log_level: str = "DEBUG"

    #openai_model: str = "gpt4o-mini"
    #openai_api_key: str = "OPENAI_API_KEY"


@cache
def get_settings():
    print("getting settings...")
    return Settings()