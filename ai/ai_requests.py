from config import Config
from openai import AzureOpenAI
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    config = Config()
    logger.info("Config loaded successfully")
except Exception as e:
    logger.error(f"Failed to load config: {str(e)}")
    raise

try:
    chat_client = AzureOpenAI(
        api_key=config.api_key,
        api_version=config.api_version,
        azure_endpoint=config.api_base,
    )
    logger.info("Successfully initialized chatAzure OpenAI client")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise


def ai_story(objects: list[str]) -> str:
    try:
        system_prompt = """
        You are a storyteller.
        You are given a list of objects.
        You need to create a story about the objects.
        The story should be a short story about the objects.
        The story should be 100 words or less.
        Keep it safe for work and feel free to be funny.
        Feel free to write a poem too. Try to keep it short.
        Make up names if needed.
        five sentences max.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Objects: {objects}"},
        ]
        response = chat_client.chat.completions.create(
            model=config.deployment_name,
            messages=messages,
            temperature=0.3,
            max_tokens=1000,
        )
        story = response.choices[0].message.content.strip()
        return story
    except Exception as e:
        logger.error(f"Error in AI story generation: {str(e)}")
        raise
