import config
import logging

config = config.Config()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_config_exists():
    assert config is not None


def test_api_type():
    assert hasattr(config, "api_type")
    api_type = config.api_type
    logger.info(f"api_type value: {api_type}")
    assert isinstance(api_type, str)


def test_deployment_name():
    assert hasattr(config, "deployment_name")
    deployment_name = config.deployment_name
    logger.info(f"deployment_name value: {deployment_name}")
    assert isinstance(deployment_name, str)


def test_api_base():
    logger.info("Testing api_base configuration")
    assert hasattr(config, "api_base")
    api_base = config.api_base
    assert isinstance(api_base, str)
    assert "hou-bootcamp-openai" in api_base, (
        "Please check the embedding_api_base value - "
        "it should contain 'hou-bootcamp-openai'"
    )


def test_api_key():
    logger.info("Testing api_key configuration")
    assert hasattr(config, "api_key")
    api_key = config.api_key
    assert isinstance(api_key, str)
    assert len(api_key) > 0


def test_embedding_api_base():
    logger.info("Testing embedding_api_base configuration")
    assert hasattr(config, "embedding_api_base")
    embedding_api_base = config.embedding_api_base
    assert isinstance(embedding_api_base, str)
    assert "accessibility-chat" in embedding_api_base, (
        "Please check the embedding_api_base value - "
        "it should contain 'accessibility-chat'"
    )


def test_embedding_api_key():
    logger.info("Testing embedding_api_key configuration")
    assert hasattr(config, "embedding_api_key")
    embedding_api_key = config.embedding_api_key
    assert isinstance(embedding_api_key, str)
    assert len(embedding_api_key) > 0
