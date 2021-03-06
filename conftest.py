import multiprocessing
import sys

import pytest


def pytest_addoption(parser):
  parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
  if config.getoption("--runslow"):
    # --runslow given in cli: do not skip slow tests
    return
  skip_slow = pytest.mark.skip(reason="need --runslow option to run")
  for item in items:
    if "slow" in item.keywords:
      item.add_marker(skip_slow)


def pytest_load_initial_conftests(args):
  if "xdist" in sys.modules:  # pytest-xdist plugin
    num = max(multiprocessing.cpu_count() / 2, 1)
    args[:] = ["-n", str(num)] + args