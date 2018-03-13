# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Run gather and train in a loop, as subprocesses.

We run as subprocesses because it gives us some isolation.
If the gather job dies more than three times, we quit entirely.
"""

import subprocess
import sys
from utils import timer
import rl_loop_9x9


def loop():
    """Run gather and train as subprocesses."""
    gather_errors = 0
    while True:
        print("==================================")
        with timer("Gather"):
            # the stupid way, do not separate data collection and training
            rl_loop_9x9.gather()
            '''# hacky
            python_dir = "/Users/zhuoyuan/miniconda3/bin/"
            gather = subprocess.call(python_dir+"python rl_loop_9x9.py gather", shell=True)
            if gather != 0:
                print("Error in gather, retrying")
                gather_errors += 1
                if gather_errors == 3:
                    print("Gathering died too many times!")
                    sys.exit(1)
                continue
            '''
        gather_errors = 0

        rl_loop_9x9.train()
        # with timer("Train"):
        #     subprocess.call("python rl_loop.py train", shell=True)

        with timer("validate"):
            subprocess.call("python rl_loop.py validate", shell=True)


if __name__ == '__main__':
    loop()
