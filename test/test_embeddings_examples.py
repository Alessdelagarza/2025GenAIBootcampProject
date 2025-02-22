import pytest
from data.embeddings import find_most_similar_frame, get_embeddings
from unittest.mock import patch
from openai import AzureOpenAI
from config import Config

config = Config()


@pytest.mark.parametrize(
    "query,expected_frame",
    [
        ("make it black and white", "grayscale"),
        ("show me what objects are in the video", "object_detection"),
        ("make it look painted", "water_color"),
        ("show temperature visualization", "heat_map"),
    ],
)
def test_specific_mappings(query, expected_frame):
    """Test that specific queries map to expected frames"""
    frame_name, similarity = find_most_similar_frame(query)
    assert frame_name == expected_frame
    assert similarity > 0.5


def test_semantic_similarity_painting():
    """Test that semantically similar painting-related queries return consistent results"""
    painting_queries = [
        "make this look like a painting",
        "transform this into artwork",
        "apply a painted effect",
        "give it an artistic look",
    ]

    results = [find_most_similar_frame(query)[0] for query in painting_queries]
    assert all(
        result == "water_color" for result in results
    ), "All painting-related queries should map to water_color effect"


def test_semantic_similarity_thermal():
    """Test that semantically similar thermal/heat-related queries return consistent results"""
    thermal_queries = [
        "show me how hot everything is",
        "display temperature visualization",
        "make it look like thermal vision",
        "heat vision effect",
    ]

    results = [find_most_similar_frame(query)[0] for query in thermal_queries]
    assert all(
        result == "heat_map" for result in results
    ), "All thermal-related queries should map to heat_map effect"


@pytest.mark.parametrize(
    "query,expected_frame",
    [
        ("make this black and white", "grayscale"),
        ("detect objects in the scene", "object_detection"),
        ("make it look painted", "water_color"),
        ("show heat vision", "heat_map"),
    ],
)
def test_confidence_scores(query, expected_frame):
    """Test that relevant queries have high confidence scores"""
    frame_name, similarity = find_most_similar_frame(query)
    assert frame_name == expected_frame
    assert similarity > 0.7, f"Low confidence ({similarity}) for query: {query}"


def test_irrelevant_queries():
    """Test that unrelated queries return lower confidence scores"""
    irrelevant_queries = [
        "what's the weather like today",
        "tell me a joke",
        "what time is it",
        "how do I make coffee",
    ]

    results = [find_most_similar_frame(query)[1] for query in irrelevant_queries]
    assert all(
        score < 0.7 for score in results
    ), "Irrelevant queries should have lower confidence scores"


# Optional: Mock test to avoid API calls during testing
@pytest.mark.optional
def test_with_mock_embeddings():
    """Test the function with mocked embeddings to avoid API calls"""
    mock_embedding = [0.1] * 1536  # Standard OpenAI embedding size

    with patch("data.embeddings.get_embeddings", return_value=mock_embedding):
        frame_name, similarity = find_most_similar_frame("test query")
        assert isinstance(frame_name, str)
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1


def test_embedding_shape():
    """Test that embeddings have the correct shape"""
    embedding = get_embeddings("test query")
    assert len(embedding) == 1536, "OpenAI embeddings should be 1536-dimensional"


def test_llm_judge_validation():
    """Use LLM to validate if the frame selections make sense"""
    test_cases = [
        "make this look like a watercolor painting",
        "show me what objects are in this scene",
        "convert this to black and white",
        "show me the heat signature",
    ]

    client = AzureOpenAI(
        api_key=config.api_key,
        api_version=config.api_version,
        azure_endpoint=config.api_base,
    )

    for query in test_cases:
        frame_name, similarity = find_most_similar_frame(query)

        # Construct prompt for LLM judgment
        prompt = f"""You are an expert judge evaluating video effect selections.

        User Query: "{query}"
        Selected Effect: "{frame_name}"
        Confidence Score: {similarity:.2f}

        Available effects and their purposes:
        - water_color: Creates artistic, painting-like effects
        - object_detection: Identifies and labels objects in the scene
        - grayscale: Converts image to black and white
        - heat_map: Shows temperature-like visualization

        Question: Is this effect selection appropriate for the user's query?
        Answer only with 'YES' or 'NO' followed by a brief explanation.
        """

        response = client.chat.completions.create(
            model=config.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )

        judgment = response.choices[0].message.content.strip().upper()
        assert judgment.startswith(
            "YES"
        ), f"LLM rejected the mapping of '{query}' to '{frame_name}'. Judgment: {judgment}"


@pytest.mark.parametrize(
    "query,invalid_frame",
    [
        ("make it look painted", "grayscale"),
        ("show me objects", "heat_map"),
        ("make it black and white", "water_color"),
    ],
)
def test_llm_judge_negative_cases(query, invalid_frame):
    """Test that LLM correctly identifies inappropriate effect selections"""
    client = AzureOpenAI(
        api_key=config.api_key,
        api_version=config.api_version,
        azure_endpoint=config.api_base,
    )

    prompt = f"""You are an expert judge evaluating video effect selections.

    User Query: "{query}"
    Selected Effect: "{invalid_frame}"

    Available effects and their purposes:
    - water_color: Creates artistic, painting-like effects
    - object_detection: Identifies and labels objects in the scene
    - grayscale: Converts image to black and white
    - heat_map: Shows temperature-like visualization

    Question: Is this effect selection appropriate for the user's query?
    Answer only with 'YES' or 'NO' followed by a brief explanation.
    """

    response = client.chat.completions.create(
        model=config.deployment_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )

    judgment = response.choices[0].message.content.strip().upper()
    assert judgment.startswith(
        "NO"
    ), f"LLM incorrectly approved the mapping of '{query}' to '{invalid_frame}'"


@pytest.mark.optional
def test_llm_judge_edge_cases():
    """Test how the system handles ambiguous or complex requests"""
    edge_cases = [
        "make it look artistic but also show the temperature",
        "detect objects and make them black and white",
        "I want to see both heat signatures and painted effects",
    ]

    client = AzureOpenAI(
        api_key=config.api_key,
        api_version=config.api_version,
        azure_endpoint=config.api_base,
    )

    for query in edge_cases:
        frame_name, similarity = find_most_similar_frame(query)

        prompt = f"""You are an expert judge evaluating video effect selections.

        User Query: "{query}"
        Selected Effect: "{frame_name}"
        Confidence Score: {similarity:.2f}

        Context: The system can only apply one effect at a time.

        Question: Given the limitation of one effect at a time, is the selected effect
        a reasonable choice for the main intent of the query?
        Answer only with 'YES' or 'NO' followed by a brief explanation.
        """

        response = client.chat.completions.create(
            model=config.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )

        judgment = response.choices[0].message.content.strip()
        print(f"Edge case: {query}\nSelected: {frame_name}\nJudgment: {judgment}\n")
