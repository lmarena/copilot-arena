import pytest
import yaml
from collections import Counter
import numpy as np
from scipy import stats
from fastapi.testclient import TestClient
from app import fastapp  # Import the existing FastAPI app instance
from src.utils import get_settings


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


def get_ground_truth_probabilities():
    """
    Extract ground truth probabilities from the YAML config file.
    Returns a dictionary of model names to their normalized probabilities.
    """
    # Read the YAML file
    config = get_settings()

    # Extract weights for active models (not commented out)
    model_weights = {
        model_name: model_info["weight"]
        for model_name, model_info in config["models"].items()
    }

    # Calculate total weight for normalization
    total_weight = sum(model_weights.values())

    # Calculate normalized probabilities
    probabilities = {
        model_name: weight / total_weight
        for model_name, weight in model_weights.items()
    }

    return probabilities


def calculate_expected_paired_probabilities(ground_truth_probs):
    """
    Calculate expected probabilities when sampling pairs without replacement.

    For each model M, its total probability is:
    P(M) = P(M selected first) + P(M selected second)
    = P(M first) + sum[P(other first) * P(M second | other first)]
    """
    models = list(ground_truth_probs.keys())
    n_models = len(models)
    adjusted_probs = {}

    for model in models:
        prob = 0
        # Probability of being selected first
        prob_first = ground_truth_probs[model]

        # Probability of being selected second
        for other_model in models:
            if other_model != model:
                # If other_model is selected first (prob_first_other),
                # then model's prob of being selected second is its weight divided by
                # sum of all weights except other_model's weight
                prob_first_other = ground_truth_probs[other_model]
                remaining_weight = sum(
                    ground_truth_probs[m] for m in models if m != other_model
                )
                prob_second_given_first = ground_truth_probs[model] / remaining_weight
                prob += prob_first_other * prob_second_given_first

        # Total probability is sum of being selected first or second
        total_prob = prob_first + prob
        adjusted_probs[model] = total_prob

    # Normalize probabilities
    total = sum(adjusted_probs.values())
    return {model: prob / total for model, prob in adjusted_probs.items()}


def test_model_distribution(fast_app, n_trials):
    """Test if the distribution of individual model selections matches expected probabilities"""
    # Get ground truth probabilities from config
    ground_truth_probs = get_ground_truth_probabilities()

    # Calculate adjusted probabilities for paired sampling
    expected_probs = calculate_expected_paired_probabilities(ground_truth_probs)

    # Collect samples - count each model individually
    selected_models = []
    for _ in range(n_trials):
        models, _, _ = fast_app.select_models(tags=[])
        selected_models.extend(models)

    # Count occurrences of each model
    model_counts = Counter(selected_models)

    # Calculate total selections (2 models per trial)
    total_selections = n_trials * 2

    # Print analysis
    print("\nModel Distribution Analysis:")
    print("\nProbability Comparison:")
    print(
        f"{'Model':<30} {'Original':<12} {'Adjusted':<12} {'Observed':<12} {'Diff %':<10}"
    )
    print("-" * 75)

    # Prepare arrays for chi-square test
    observed_freqs = []
    expected_freqs = []

    for model in sorted(ground_truth_probs.keys()):
        original_prob = ground_truth_probs[model]
        expected_prob = expected_probs[model]
        observed_count = model_counts[model]
        observed_prob = observed_count / total_selections
        diff_percent = ((observed_prob - expected_prob) / expected_prob) * 100

        print(
            f"{model:<30} {original_prob:>11.4f} {expected_prob:>11.4f} "
            f"{observed_prob:>11.4f} {diff_percent:>+9.1f}%"
        )

        # Add to arrays for chi-square test
        expected_freqs.append(expected_prob * total_selections)
        observed_freqs.append(observed_count)

    # Perform chi-square test
    chi2, p_value = stats.chisquare(observed_freqs, expected_freqs)

    print("\nStatistical Analysis:")
    print(f"Total selections: {total_selections}")
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Assert that p-value is above threshold
    assert (
        p_value > 0.05
    ), f"Distribution of selected models differs significantly from expected (p={p_value:.4f})"


def test_tag_filtering(fast_app):
    """Test if model selection respects tag filtering"""
    # Test with a specific tag
    test_tag = list(fast_app.tag_to_models.keys())[0]  # Get first available tag
    tagged_models = fast_app.tag_to_models[test_tag]

    # Sample multiple times with the tag
    for _ in range(100):
        models, client1, client2 = fast_app.select_models(tags=[test_tag])
        # Check if selected models have the required tag
        assert all(
            model in tagged_models for model in models
        ), f"Selected models {models} don't all have tag {test_tag}"


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
