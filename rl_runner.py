
import subprocess
import argh
from utils import timer

def loop(logdir=None):
    gather_errors = 0
    while True:
        print("==================================")
        with timer("Gather"):
            gather = subprocess.call("python rl_loop.py gather".split())
            if gather != 0:
                print("Error in gather, retrying")
                gather_errors+=1
                if gather_errors == 3:
                    print("Gathering died too many times!")
                    sys.exit(1)
                continue
        gather_errors = 0

        with timer("Train"):
            subprocess.call( ("python rl_loop.py train --logdir=%s" % logdir).split())


if __name__ == '__main__':
    argh.dispatch_command(loop)
