from docassist.config import Settings


def test_defaults_point_at_public_api_and_opus(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8787")
    settings = Settings(_env_file=None)
    # The unprefixed variable must not leak into our settings.
    assert settings.anthropic_base_url == "https://api.anthropic.com"
    assert settings.model == "claude-opus-5-5"
    assert settings.anthropic_api_key is None


def test_prefixed_env_overrides(monkeypatch):
    monkeypatch.setenv("DOCASSIST_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("DOCASSIST_ANTHROPIC_API_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert settings.model == "claude-sonnet-5-5"
    assert settings.anthropic_api_key.get_secret_value() == "sk-test"
    assert "sk-test" not in repr(settings)
