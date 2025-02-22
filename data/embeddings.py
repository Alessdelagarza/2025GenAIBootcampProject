import logging
import numpy as np
import pandas as pd
import os
import filecmp
from ast import literal_eval
from config import Config
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

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
    df = pd.read_csv("data/frame_descriptions.csv")
    logger.info("Successfully loaded frame descriptions data")
except Exception as e:
    logger.error(f"Error loading frame descriptions data: {str(e)}")
    raise

try:
    embeddings_client = AzureOpenAI(
        api_key=config.embedding_api_key,
        api_version=config.embedding_api_version,
        azure_endpoint=config.embedding_api_base,
    )
    logger.info("Successfully initialized Azure OpenAI client")
except Exception as e:
    logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
    raise


@lru_cache(maxsize=1000)
def get_embeddings(text: str) -> tuple:
    try:
        response = embeddings_client.embeddings.create(
            input=text, model=config.embedding_deployment_name
        )
        return tuple(response.data[0].embedding)
    except Exception as e:
        logger.error(f"Error getting embeddings: {str(e)}")
        raise


temp_csv = "data/frame_descriptions_temp.csv"
df.to_csv(temp_csv, index=False)

embeddings_file = "data/frame_descriptions_embeddings.csv"
regenerate_embeddings = True

if os.path.exists(embeddings_file):
    if filecmp.cmp(temp_csv, "data/frame_descriptions.csv"):
        try:
            df = pd.read_csv(embeddings_file)
            regenerate_embeddings = False
            logger.info(
                "Using existing embeddings - no changes detected in descriptions"
            )
        except Exception as e:
            logger.error(f"Error reading existing embeddings file: {str(e)}")
            regenerate_embeddings = True
    else:
        logger.info("Changes detected in descriptions - regenerating embeddings")

if regenerate_embeddings:
    try:
        df["embedding"] = df["description"].apply(lambda x: list(get_embeddings(x)))
        df.to_csv(embeddings_file, index=False)
        logger.info("Successfully created and saved new embeddings")
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise

# Clean up temp file
try:
    os.remove(temp_csv)
except Exception as e:
    logger.warning(f"Failed to remove temporary file: {str(e)}")


def find_most_similar_frame(query_text: str) -> tuple[str, float]:
    try:
        query_embedding = list(get_embeddings(query_text))

        if isinstance(df["embedding"].iloc[0], str):
            df["embedding"] = df["embedding"].apply(literal_eval)

        similarities = cosine_similarity([query_embedding], df["embedding"].tolist())[0]
        most_similar_idx = np.argmax(similarities)
        return df["frame_name"].iloc[most_similar_idx], similarities[most_similar_idx]

    except Exception as e:
        logger.error(f"Error finding most similar frame: {str(e)}")
        raise


def explain_frame_selection(
    query_text: str, frame_name: str, similarity_score: float
) -> str:
    try:
        frame_description = df[df["frame_name"] == frame_name]["description"].iloc[0]
        explanation = (
            f"Based on your request '{query_text}', I selected the {frame_name} frame. "
            f"This frame was chosen because {frame_description.lower()} "
            f"""The similarity score of {similarity_score:.2f} indicates a strong match between
            your request and this frame's capabilities."""
        )
        return explanation
    except Exception as e:
        logger.error(f"Error explaining frame selection: {str(e)}")
        raise


# Example usage:
# query = "make this video look like it was painted by an artist"
# frame_name, similarity = find_most_similar_frame(query)
# explanation = explain_frame_selection(query, frame_name, similarity)
# print(f"Most similar frame: {frame_name}")
# print(f"Explanation: {explanation}")
