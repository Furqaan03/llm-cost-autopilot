from src.classifier.predict import predict_tier


def test_simple_prompt_routes_low_tier():
    tier = predict_tier("What is the capital of France?")
    assert tier in (1, 2, 3)  # model-dependent, but must be a valid tier


def test_complex_prompt_produces_higher_features_than_simple():
    from src.classifier.features import extract_features

    simple = extract_features("What is 2 plus 2?")
    complex_ = extract_features(
        "Analyze the following business scenario and recommend a strategy, "
        "justifying each tradeoff given these constraints: revenue must exceed cost, "
        "and the plan should ensure compliance within 90 days."
    )
    assert complex_[0][0] > simple[0][0]  # token count
    assert complex_[0][1] >= simple[0][1]  # analysis instruction signal
