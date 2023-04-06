python_sources()

pex_binary(
  name="train",
  entry_point="train.py",
)

pex_binary(
    name="run_train",
    entry_point="composer.cli.launcher:main",
    execution_mode="venv",
    dependencies=[
      ":train",
      "third_party:reqs#setuptools",
      "third_party:reqs#imageio",
    ]
)