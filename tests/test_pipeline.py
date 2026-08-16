from moto_pricing.pipeline import run_full_pipeline


def test_run_full_pipeline_end_to_end(tmp_path):
    result = run_full_pipeline(
        n_policies=4_000, n_prospects=1_500, seed=99, outputs_dir=tmp_path,
    )

    assert set(result.model_comparison["model"]) == {"glm", "gbm", "ensemble"}
    assert result.model_comparison["gini"].between(-1, 1).all()
    assert not result.frequency_relativities.empty
    assert set(result.funnel["stage"]) == {"Quoted", "Bound", "Renewed"}
    assert result.funnel["count"].is_monotonic_decreasing

    assert (tmp_path / "pipeline_report.md").exists()
    assert (tmp_path / "model_comparison.csv").exists()
    assert (tmp_path / "plots" / "quote_funnel.png").exists()
