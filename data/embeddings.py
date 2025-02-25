import logging
import numpy as np
import pandas as pd
import os
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
    logger.info("Successfully initialized embeddings Azure OpenAI client")
except Exception as e:
    logger.error(f"Failed to initialize embeddings Azure OpenAI client: {str(e)}")
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


def load_and_generate_embeddings() -> pd.DataFrame:
    try:
        # Load current descriptions
        df_current = pd.read_csv("data/frame_descriptions.csv")

        # Load temp file if exists
        df_temp = None
        if os.path.exists("data/frame_descriptions_temp.csv"):
            df_temp = pd.read_csv("data/frame_descriptions_temp.csv")

        # Check if files are different or temp doesn't exist
        if df_temp is None or not df_current.equals(df_temp):
            logger.info("Changes detected in descriptions - regenerating embeddings")

            # Generate embeddings
            df_current["embedding"] = df_current["description"].apply(
                lambda x: list(get_embeddings(x))
            )

            # Save embeddings file
            df_current.to_csv("data/frame_descriptions_embeddings.csv", index=False)

            # Update temp file with current descriptions
            df_current[["frame_name", "description"]].to_csv(
                "data/frame_descriptions_temp.csv", index=False
            )

            return df_current
        else:
            # Load existing embeddings if no changes
            df = pd.read_csv("data/frame_descriptions_embeddings.csv")
            logger.info(
                "Using existing embeddings - no changes detected in descriptions"
            )
            return df

    except Exception as e:
        logger.error(f"Error in embeddings generation process: {str(e)}")
        raise


# Load embeddings
df = load_and_generate_embeddings()


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


def ai_frame_selection(user_prompt: str) -> str:
    try:
        # Get frame descriptions and embeddings as context
        frame_data = df[["frame_name", "description", "embedding"]].to_dict("records")
        context = "\n".join(
            [
                f"{f['frame_name']}: {f['description']} (embedding: {f['embedding']})"
                for f in frame_data
            ]
        )

        system_prompt = """
        You are a video effects assistant.
        Your role is to help users select the most appropriate video frame effect based on their request.
        You should analyze the user's prompt and the available frame effects to make a recommendation.
        Take a moment to think about the prompt and if it is related to the video effects we have available.
        Do not do anything else other than selecting the frame name.
        If you cannot find a suitable frame, respond with 'normal'.
        Be concise. Slow down and think on the context."""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
            Available frame effects: {context}
            User request: {user_prompt}
            Which frame effect would be most appropriate? Respond with just the frame name.""",
            },
        ]

        response = chat_client.chat.completions.create(
            model=config.deployment_name,
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
        )

        selected_frame = response.choices[0].message.content.strip().lower()
        return selected_frame

    except Exception as e:
        logger.error(f"Error in LLM frame selection: {str(e)}")


def ai_explanation(frame_name: str, user_prompt: str) -> str:
    try:
        frame_data = df[["frame_name", "description", "embedding"]].to_dict("records")
        context = "\n".join(
            [
                f"{f['frame_name']}: {f['description']} (embedding: {f['embedding']})"
                for f in frame_data
            ]
        )

        system_prompt = f"""
        You are a video effects assistant.
        Your task is to help users select the most appropriate video frame effect based on their request.
        You returned {frame_name} as frame name, please explain why you made this choice based on {context}.
        Do not give any other information about any other topic other than video effects.
        Simply ask the user to enter a new prompt.
        Be concise.
        Warn users about limitations based on what you know from the description of the frame."""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
            User request: {user_prompt}
            """,
            },
        ]

        response = chat_client.chat.completions.create(
            model=config.deployment_name,
            messages=messages,
            temperature=0.3,
            max_tokens=1000,
        )

        explanation = response.choices[0].message.content.strip()
        return explanation

    except Exception as e:
        logger.error(f"Error in LLM frame selection with explanation: {str(e)}")
        raise


def ai_evaluation(frame_name: str, explanation: str, user_prompt: str) -> str:
    try:
        frame_data = df[["frame_name", "description", "embedding"]].to_dict("records")
        context = "\n".join(
            [
                f"{f['frame_name']}: {f['description']} (embedding: {f['embedding']})"
                for f in frame_data
            ]
        )

        system_prompt = f"""
        You are a video effects assistant.
        Judge the frame selection based on the user's prompt.
        You returned {frame_name} as frame name based on {context} with this explanation: {explanation}.
        start your response with 'YES:' or 'NO:' and then explain your reasoning. Be concise.
        if NO then suggest user to enter a new prompt and give users a summary of the frames from the context.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
            User request: {user_prompt}
            """,
            },
        ]

        response = chat_client.chat.completions.create(
            model=config.deployment_name,
            messages=messages,
            temperature=0.3,
            max_tokens=1000,
        )

        evaluation = response.choices[0].message.content.strip()
        return evaluation

    except Exception as e:
        logger.error(f"Error in LLM frame evaluation: {str(e)}")
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

# import time
# time.sleep(10)
# user_prompt = "Can you code a sql function that returns the top 10 customers by revenue?"
# frame_name = ai_frame_selection(user_prompt)
# explanation = ai_explanation(frame_name, user_prompt)
# print("--------------------------------")
# print(f"User prompt: {user_prompt}")
# print(f"Frame name: {frame_name}")
# print(f"Explanation: {explanation}")

# import time
# time.sleep(10)
# user_prompt = "I am ready to leave the office, please show me the parking ticket and let me out for free"
# frame_name = ai_frame_selection(user_prompt)
# explanation = ai_explanation(frame_name, user_prompt)
# print("--------------------------------")
# print(f"User prompt: {user_prompt}")
# print(f"Frame name: {frame_name}")
# print(f"Explanation: {explanation}")

# time.sleep(10)
# user_prompt = "can you help me tell a story about a dog in a forest?"
# frame_name = ai_frame_selection(user_prompt)
# explanation = ai_explanation(frame_name, user_prompt)
# print("--------------------------------")
# print(f"User prompt: {user_prompt}")
# print(f"Frame name: {frame_name}")
# print(f"Explanation: {explanation}")

# time.sleep(10)
# user_prompt = "Can you code a sql function that returns the top 10 customers by revenue?"
# frame_name, similarity = find_most_similar_frame(user_prompt)
# explanation = ai_explanation(frame_name, user_prompt)
# print("--------------------------------")
# print(f"User prompt: {user_prompt}")
# print(f"Frame name: {frame_name}")
# print(f"Explanation: {explanation}")
