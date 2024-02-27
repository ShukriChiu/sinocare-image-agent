import os

from langchain_openai import AzureChatOpenAI, ChatOpenAI


def get_azure_llm_config(deployment_name, api_key_env, api_base_env):
    AZURE_OPENAI_API_KEY = os.environ[api_key_env]
    AZURE_OPENAI_ENDPOINT = os.environ[api_base_env]
    AZURE_OPENAI_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
    HELICONE_API = os.environ["HELICONE_API"]
    helicone_headers = {
        "Helicone-OpenAI-Api-Base": f"{AZURE_OPENAI_ENDPOINT}",
        "Helicone-Auth": f"Bearer {HELICONE_API}",
        "Helicone-Cache-Enabled": "false",
        "api-key": f"{AZURE_OPENAI_API_KEY}",
    }
    return {
        "model": "gpt-4",
        "model_version": "1106-Preview",
        "api_key": AZURE_OPENAI_API_KEY,
        "base_url": f"https://oai.hconeai.com/openai/deployments/{deployment_name}",
        "api_version": AZURE_OPENAI_API_VERSION,
        "temperature": 0.3,
        "default_headers": helicone_headers,
        "default_query": {"api-version": AZURE_OPENAI_API_VERSION},
        "streaming": True,
    }


def initialize_azure_gpt4() -> AzureChatOpenAI:
    # 900 RPM
    llm_india = AzureChatOpenAI(
        **get_azure_llm_config(
            "sinocareGPT-india-gpt4",
            "AZURE_OPENAI_API_KEY_INDIA",
            "AZURE_OPENAI_ENDPOINT_INDIA",
        )
    )
    # 480 RPM
    llm_aus = AzureChatOpenAI(
        **get_azure_llm_config(
            "sinocareGPT-aus-gpt4",
            "AZURE_OPENAI_API_KEY_AUS",
            "AZURE_OPENAI_ENDPOINT_AUS",
        )
    )
    # 480 RPM
    llm_eastus1 = AzureChatOpenAI(
        **get_azure_llm_config(
            "sinocareGPT-eastus-gpt4",
            "AZURE_OPENAI_API_KEY_EASTUS1",
            "AZURE_OPENAI_ENDPOINT_EASTUS1",
        )
    )
    # 480 RPM
    llm_westus = AzureChatOpenAI(
        **get_azure_llm_config(
            "sinocareGPT-westus-gpt4",
            "AZURE_OPENAI_API_KEY_WESTUS",
            "AZURE_OPENAI_ENDPOINT_WESTUS",
        )
    )

    # 480 RPM
    llm_scus = AzureChatOpenAI(
        **get_azure_llm_config(
            "sinocareGPT-scus-gpt4",
            "AZURE_OPENAI_API_KEY_SCUS",
            "AZURE_OPENAI_ENDPOINT_SCUS",
        )
    )
    # return llm_india.with_fallbacks([llm_aus, llm_eastus1, llm_westus])
    return llm_scus


def initialize_azure_gpt35() -> AzureChatOpenAI:
    # 720 RPM
    llm_india = AzureChatOpenAI(
        **get_azure_llm_config(
            "sinocareGPT-india-gpt35",
            "AZURE_OPENAI_API_KEY_INDIA",
            "AZURE_OPENAI_ENDPOINT_INDIA",
        )
    )
    # 720 RPM
    llm_aus = AzureChatOpenAI(
        **get_azure_llm_config(
            "sinocareGPT-aus-gpt35",
            "AZURE_OPENAI_API_KEY_AUS",
            "AZURE_OPENAI_ENDPOINT_AUS",
        )
    )
    # 1800 RPM
    llm_eastus1 = AzureChatOpenAI(
        **get_azure_llm_config(
            "sinocareGPT-eastus-gpt35",
            "AZURE_OPENAI_API_KEY_EASTUS1",
            "AZURE_OPENAI_ENDPOINT_EASTUS1",
        )
    )
    # 720 RPM
    llm_westus = AzureChatOpenAI(
        **get_azure_llm_config(
            "sinocareGPT-westus-gpt35",
            "AZURE_OPENAI_API_KEY_WESTUS",
            "AZURE_OPENAI_ENDPOINT_WESTUS",
        )
    )
    return llm_india.with_fallbacks([llm_aus, llm_eastus1, llm_westus])


def initialize_openai_gpt4() -> ChatOpenAI:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HELICONE_API_KEY = os.getenv("HELICONE_API_KEY")
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4-turbo-preview",
        temperature=0.3,
        model_kwargs={"extra_headers": {"Helicone-Auth": f"Bearer {HELICONE_API_KEY}"}},
        base_url="https://oai.hconeai.com/v1",
        streaming=True,
    )
    return llm


def initialize_openai_gpt35() -> ChatOpenAI:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HELICONE_API_KEY = os.getenv("HELICONE_API_KEY")
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-3.5-turbo-0125",
        temperature=0.3,
        model_kwargs={"extra_headers": {"Helicone-Auth": f"Bearer {HELICONE_API_KEY}"}},
        base_url="https://oai.hconeai.com/v1",
        streaming=True,
    )
    return llm
