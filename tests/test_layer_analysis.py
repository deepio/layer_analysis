import toml

import layer_analysis


def test_version():
  """
  Make sure the python version in the `TOML` file matches the version in `__init__.py`.
  """
  with open("pyproject.toml") as f:
    data = toml.loads(f.read())
  assert layer_analysis.__version__ == data["tool"]["poetry"]["version"]

def test_set_values():
  layer_analysis.set_defaults({
    "_VALIDATION_SPLIT_": 0.3,
    "_BATCH_SIZE_": 17,
    "_EPOCHS_": 35,
    "_SAMPLES_PER_CLASS_": 3500,
    "_SPEED_FACTOR_": 190.,
  })
  # default is 0.2
  assert layer_analysis._VALIDATION_SPLIT_ == 0.3
  # default is 16
  assert layer_analysis._BATCH_SIZE_ == 17
  # default is 15
  assert layer_analysis._EPOCHS_ == 35
  # default is 2000
  assert layer_analysis._SAMPLES_PER_CLASS_ == 3500
  # default is 100.
  assert layer_analysis._SPEED_FACTOR_ == 190.
