# python built-ins
import logging
from typing import Any, Optional, Union


# external packages
from matplotlib import pyplot as plt

plt.set_loglevel("error")
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("PIL").disabled = True
