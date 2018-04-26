import os

import sgf
from tensorflow import gfile
from tqdm import tqdm

import dual_net
import shipname
from gtp_wrapper import MCTSPlayer
import sgf_wrapper
from utils import parse_game_result, logged_timer


def parse_sgf(sgf_path):
    with open(sgf_path) as f:
        sgf_contents = f.read()

    return zip(*[(p.position, p.next_move, p.result)
                 for p in sgf_wrapper.replay_sgf(sgf_contents)])


def check_year(props, year):
    if year is None:
        return True
    if props.get('DT') is None:
        return False

    try:
        # Most sgf files in this database have dates of the form
        #"2005-01-15", but there are some rare exceptions like
        #"Broadcasted on 2005-01-15.
        year_sgf = int(props.get('DT')[0][:4])
    except:
        return False
    return year_sgf >= year


def check_komi(props, komi_str):
    if komi_str is None:
        return True
    if props.get('KM') is None:
        return False
    return props.get('KM')[0] == komi_str


def find_and_filter_sgf_files(base_dir, min_year=None, komi=None):
    sgf_files = []
    print("Finding all sgf files in {} with year >= {} and komi = {}".format(
        base_dir, min_year, komi))
    count = 0
    for dirpath, dirnames, filenames in tqdm(os.walk(base_dir)):
        for filename in filenames:
            count += 1
            if count % 5000 == 0:
                print("Parsed {}, Found {}".format(count, len(sgf_files)))
            if filename.endswith('.sgf'):
                path = os.path.join(dirpath, filename)
                with open(path) as f:
                    sgf_contents = f.read()
                props = sgf_wrapper.get_sgf_root_node(sgf_contents).properties
                if check_year(props, min_year) and check_komi(props, komi):
                    sgf_files.append(path)
    print("Found {} sgf files matching filters".format(len(sgf_files)))
    return sgf_files


def get_model_paths(model_dir):
    '''Returns all model paths in the model_dir.'''
    all_models = gfile.Glob(os.path.join(model_dir, '*.meta'))
    model_filenames = [os.path.basename(m) for m in all_models]
    model_numbers_names = [
        (shipname.detect_model_num(m), shipname.detect_model_name(m))
        for m in model_filenames]
    model_names = sorted(model_numbers_names)
    return [os.path.join(model_dir, name[1]) for name in model_names]


def load_player(model_path):
    print("Loading weights from %s ... " % model_path)
    with logged_timer("Loading weights from %s ... " % model_path):
        network = dual_net.DualNetwork(model_path)
        network.name = os.path.basename(model_path)
    player = MCTSPlayer(network, verbosity=2)
    return player


def restore_params(model_path, player):
    with player.network.sess.graph.as_default():
        player.network.initialize_weights(model_path)
