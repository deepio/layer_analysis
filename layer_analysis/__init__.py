__version__ = '1.0a0'

##############################
#      DEFAULT SETTINGS
##############################
_VALIDATION_SPLIT_ = 0.2
_BATCH_SIZE_ = 16
_EPOCHS_ = 15
_SAMPLES_PER_CLASS_ = 2000
_SPEED_FACTOR_ = 100.
##############################

from . import utils as utils


def set_defaults(data: dict) -> dict:
  """
  Sanitize keys for only those which match:
    * First character is _
    * Second character is not _
  Then set them.
  """
  data = {k:v for (k, v) in data.items() if k[0] == "_" and k[1] != "_"}

  for k, v in data.items():
    globals()[k] = v

  # Return the dict. if needed.
  return data
