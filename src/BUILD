python_sources()

pex_binary(
    name="run_train",
    entry_point="composer.cli.launcher:main",
    dependencies=[
      "third_party:reqs#setuptools",
      "third_party:reqs#torch",
      "third_party:reqs#torchvision",
    ]
)