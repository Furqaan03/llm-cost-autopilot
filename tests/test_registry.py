from src.registry import get_model


def test_cost_calculation():
    model = get_model("gpt-4o-mini")
    cost = model.cost_for(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == model.cost_per_1m_input + model.cost_per_1m_output


def test_local_model_is_free():
    model = get_model("llama3-local")
    assert model.cost_for(10_000, 10_000) == 0.0


def test_unknown_model_raises():
    import pytest

    with pytest.raises(KeyError):
        get_model("not-a-real-model")
