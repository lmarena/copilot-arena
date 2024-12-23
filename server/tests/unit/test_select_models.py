import pytest
import yaml
from collections import Counter
import numpy as np
from scipy import stats
from fastapi.testclient import TestClient
from app import fastapp  # Import the existing FastAPI app instance
from src.utils import get_settings, get_models_by_tags


@pytest.fixture(scope="session")
def fast_app():
    """
    Get the FastAPIApp instance from the existing app
    """
    return fastapp


@pytest.fixture(scope="session")
def n_trials():
    """Number of trials for distribution testing"""
    return 300000


def get_ground_truth_probabilities(is_edit=False):
    """
    Extract ground truth probabilities from the YAML config file.
    Returns a dictionary of model names to their normalized probabilities.
    """
    # Read the YAML file
    config = get_settings()

    # Get the distribution for the correct request type
    dist_key = "edit" if is_edit else "autocomplete"
    distribution = config.get("dist", {}).get(dist_key, {})

    # Initialize probability counters for each model
    model_probs = {}

    # Calculate probability for each model based on pair distributions
    for pair_str, prob in distribution.items():
        model1, model2 = pair_str.split("|||")

        # Add probability for first model
        if model1 not in model_probs:
            model_probs[model1] = 0
        model_probs[model1] += prob / 2  # Divide by 2 because order is randomized

        # Add probability for second model
        if model2 not in model_probs:
            model_probs[model2] = 0
        model_probs[model2] += prob / 2  # Divide by 2 because order is randomized

    # Normalize probabilities
    total_prob = sum(model_probs.values())
    if total_prob > 0:
        return {model: prob / total_prob for model, prob in model_probs.items()}
    return {}


def test_model_distribution(fast_app, n_trials):
    """Test if the distribution of individual model selections matches expected probabilities"""
    # Test both autocomplete and edit distributions
    for is_edit in [False, True]:
        tags = ["edit"] if is_edit else []
        request_type = "edit" if is_edit else "autocomplete"

        # Get ground truth probabilities from config
        expected_probs = get_ground_truth_probabilities(is_edit)
        if not expected_probs:
            continue  # Skip if no distribution defined for this type

        # Collect samples
        selected_models = []
        for _ in range(n_trials):
            models, _, _ = fast_app.select_models(tags=tags)
            selected_models.extend(models)

        # Count occurrences of each model
        model_counts = Counter(selected_models)
        total_selections = n_trials * 2

        # Print analysis
        print(f"\nModel Distribution Analysis ({request_type}):")
        print("\nProbability Comparison:")
        print(f"{'Model':<30} {'Expected':<12} {'Observed':<12} {'Diff %':<10}")
        print("-" * 65)

        # Prepare arrays for chi-square test
        observed_freqs = []
        expected_freqs = []

        for model in sorted(expected_probs.keys()):
            expected_prob = expected_probs[model]
            observed_count = model_counts[model]
            observed_prob = observed_count / total_selections
            diff_percent = ((observed_prob - expected_prob) / expected_prob) * 100

            print(
                f"{model:<30} {expected_prob:>11.4f} {observed_prob:>11.4f} "
                f"{diff_percent:>+9.1f}%"
            )

            expected_freqs.append(expected_prob * total_selections)
            observed_freqs.append(observed_count)

        # Perform chi-square test
        chi2, p_value = stats.chisquare(observed_freqs, expected_freqs)

        print("\nStatistical Analysis:")
        print(f"Total selections: {total_selections}")
        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"P-value: {p_value:.4f}")

        # Assert that p-value is above threshold
        assert p_value > 0.05, (
            f"Distribution of selected models for {request_type} differs significantly "
            f"from expected (p={p_value:.4f})"
        )


def test_tag_filtering(fast_app):
    """Test if model selection respects tag filtering"""
    # Test with edit tag
    edit_models = get_models_by_tags(["edit"], fast_app.models, fast_app.tag_to_models)
    for _ in range(100):
        models, _, _ = fast_app.select_models(tags=["edit"])
        assert all(
            model in edit_models for model in models
        ), f"Selected models {models} don't all have edit tag"

    # Test with autocomplete (no edit tag)
    autocomplete_models = get_models_by_tags(
        [], fast_app.models, fast_app.tag_to_models
    )
    for _ in range(100):
        models, _, _ = fast_app.select_models(tags=[])
        assert all(
            model in autocomplete_models for model in models
        ), f"Selected models {models} not in available models"


def test_different_models(fast_app):
    """Test if select_models always returns two different models"""
    for _ in range(100):
        models, _, _ = fast_app.select_models(tags=[])
        assert len(set(models)) == 2, f"Selected models {models} are not unique"


def test_empty_tags_uses_all_models(fast_app):
    """Test if empty tags list uses all available models"""
    all_models = set()
    n_trials = 1000

    # Run multiple trials to ensure we see all possible models
    for _ in range(n_trials):
        models, _, _ = fast_app.select_models(tags=[])
        all_models.update(models)

    # Check if we've seen all available models
    assert all_models == set(
        fast_app.models
    ), f"Not all models were selected. Missing: {set(fast_app.models) - all_models}"


def test_model_client_mapping(fast_app):
    """Test if returned clients correspond to selected models"""
    for _ in range(100):
        models, client1, client2 = fast_app.select_models(tags=[])

        # Check if clients match their respective models
        assert (
            models[0] in client1.models
        ), f"Client 1 doesn't support model {models[0]}"
        assert (
            models[1] in client2.models
        ), f"Client 2 doesn't support model {models[1]}"


def test_model_position_distribution(fast_app, n_trials):
    """Test if each model appears roughly equally often in first and second position"""
    # Track positions for each model
    position_counts = {}  # {model: [first_position_count, second_position_count]}

    # Collect samples
    for _ in range(n_trials):
        models, _, _ = fast_app.select_models(tags=[])

        # Initialize counters for new models
        for model in models:
            if model not in position_counts:
                position_counts[model] = [0, 0]

        # Count positions (index 0 for first position, 1 for second position)
        position_counts[models[0]][0] += 1
        position_counts[models[1]][1] += 1

    # Print and analyze results
    print("Position Distribution Analysis:")
    print(f"{'Model':<30} {'First Pos %':<12} {'Second Pos %':<12} {'Diff %':<10}")
    print("-" * 65)

    # For each model, check if the distribution is within 2% of 50%
    for model in sorted(position_counts.keys()):
        first_count = position_counts[model][0]
        second_count = position_counts[model][1]
        total_count = first_count + second_count

        if total_count == 0:
            continue

        first_percent = (first_count / total_count) * 100
        second_percent = (second_count / total_count) * 100
        diff_percent = first_percent - second_percent

        print(
            f"{model:<30} {first_percent:>11.1f} {second_percent:>11.1f} "
            f"{diff_percent:>+9.1f}"
        )

        # Assert that the distribution is within 2% of 50%
        assert abs(diff_percent) <= 2, (
            f"Model {model} shows significant position bias "
            f"(first={first_percent:.1f}%, second={second_percent:.1f}%)"
        )
