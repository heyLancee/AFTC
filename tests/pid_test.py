import os
import sys

module_path = os.path.abspath('./')
if module_path not in sys.path:
    sys.path.append(module_path)

from controllers import SunPointController
from eval_model import eval_pid


if __name__ == '__main__':
    pid = SunPointController(0.2, 5)
    path = "result.csv"
    eval_pid(pid, "SunPointFaultSatellite", 10, path=path, is_plot=True)
    
