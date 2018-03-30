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

"""Runs a RL loop locally. Mostly for integration testing purposes.

A successful run will bootstrap, selfplay, gather, and start training for
a while. You should see the combined_cost variable drop steadily, and ideally
overfit to a near-zero loss.
"""

import os
import tempfile

import preprocessing
import dual_net
import go
import main
from tensorflow import gfile
import subprocess


def rl_loop():
    """Run the reinforcement learning loop

    This is meant to be more of an integration test than a realistic way to run
    the reinforcement learning.
    """
    # monkeypatch the hyperparams so that we get a quickly executing network.
    dual_net.get_default_hyperparams = lambda **kwargs: {
        'k': 8, 'fc_width': 16, 'num_shared_layers': 1, 'l2_strength': 1e-4, 'momentum': 0.9}

    dual_net.TRAIN_BATCH_SIZE = 16
    dual_net.EXAMPLES_PER_GENERATION = 64

    #monkeypatch the shuffle buffer size so we don't spin forever shuffling up positions.
    preprocessing.SHUFFLE_BUFFER_SIZE = 1000

    with tempfile.TemporaryDirectory() as base_dir:
        working_dir = os.path.join(base_dir, 'models_in_training')
        model_save_path = os.path.join(base_dir, 'models', '000000-bootstrap')
        next_model_save_file = os.path.join(base_dir, 'models', '000001-nextmodel')
        selfplay_dir = os.path.join(base_dir, 'data', 'selfplay')
        model_selfplay_dir = os.path.join(selfplay_dir, '000000-bootstrap')
        gather_dir = os.path.join(base_dir, 'data', 'training_chunks')
        holdout_dir = os.path.join(
            base_dir, 'data', 'holdout', '000000-bootstrap')
        sgf_dir = os.path.join(base_dir, 'sgf', '000000-bootstrap')
        os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)

        print("Creating random initial weights...")
        main.bootstrap(working_dir, model_save_path)
        print("Playing some games...")
        # Do two selfplay runs to test gather functionality
        main.selfplay(
            load_file=model_save_path,
            output_dir=model_selfplay_dir,
            output_sgf=sgf_dir,
            holdout_pct=0,
            readouts=10)
        main.selfplay(
            load_file=model_save_path,
            output_dir=model_selfplay_dir,
            output_sgf=sgf_dir,
            holdout_pct=0,
            readouts=10)
        # Do one holdout run to test validation
        main.selfplay(
            load_file=model_save_path,
            holdout_dir=holdout_dir,
            output_dir=model_selfplay_dir,
            output_sgf=sgf_dir,
            holdout_pct=100,
            readouts=10)

        print("See sgf files here?")
        sgf_listing = subprocess.check_output(["ls", "-l", sgf_dir + "/full"])
        print(sgf_listing.decode("utf-8"))

        print("Gathering game output...")
        main.gather(input_directory=selfplay_dir, output_directory=gather_dir)
        print("Training on gathered game data...")
        main.train(working_dir, gather_dir, next_model_save_file, generation_num=1)
        print("Trying validate on 'holdout' game...")
        main.validate(working_dir, holdout_dir)
        print("Verifying that new checkpoint is playable...")
        main.selfplay(
            load_file=next_model_save_file,
            holdout_dir=holdout_dir,
            output_dir=model_selfplay_dir,
            output_sgf=sgf_dir,
            readouts=10)



if __name__ == '__main__':
    rl_loop()
