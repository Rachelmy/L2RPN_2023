"""
originally from 
https://github.com/AlibabaResearch/l2rpn-wcci-2022
modified by
Shourya Bose, Qiuling Yang, Yu Zhang
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .optimCVXPY import OPTIMCVXPY_CONFIG, OptimCVXPY


def make_agent(env, this_directory_path):

    config = OPTIMCVXPY_CONFIG.copy()
    action_space_path = this_directory_path
    time_step = 1

    agent = OptimCVXPY(
        env, 
        env.action_space, 
        action_space_path=action_space_path, 
        config=config, 
        time_step=time_step, 
        verbose=1
        )

    return agent