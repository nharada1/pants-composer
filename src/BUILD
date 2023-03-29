python_sources()

pex_binary(
    name="run_train",
    entry_point="composer.cli.launcher:main",
    execution_mode="venv",
    dependencies=[
      "third_party:reqs#setuptools",
      "third_party:reqs#torch",
      "third_party:reqs#torchvision",
    ]
)