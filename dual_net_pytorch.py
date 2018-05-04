import numpy as np

import torch
from torch.autograd import Variable

from df_model3 import Model_PolicyValue

import features
import symmetries
import go


class DualNetwork():
    def __init__(self, save_file):
        self.save_file = save_file
        self.model = Model_PolicyValue({}, {})
        self.model.eval()
        self.model.set_volatile(True)
        self.replace_prefix = [['resnet.module', 'resnet']]
        self.initialize_weights(save_file)
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model = self.model.cuda()

    def initialize_weights(self, save_file):
        replace_prefix = [['resnet.module', 'resnet']]
        self.model.load(save_file, [], replace_prefix)

    def run(self, position, use_random_symmetry=True):
        probs, values = self.run_many([position],
                                      use_random_symmetry=use_random_symmetry)
        return probs[0], values[0]

    def run_many(self, positions, use_random_symmetry=True):
        processed = list(map(features.extract_features, positions))
        # print(processed[0].shape)
        if use_random_symmetry:
            syms_used, processed = symmetries.randomize_symmetries_feat(
                processed)
        # processed: list [] of (18, N, N)
        processed = [np.reshape(item, (1, 18, go.N, go.N)) for item in processed]
        processed = np.concatenate(processed, axis=0).astype(np.float32)
        if len(processed.shape) == 3:
            processed = np.expand_dims(processed, 0)

        batch = torch.from_numpy(processed)
        if self.cuda:
            batch = batch.cuda()
        outputs = self.model(batch)
        probabilities, value = outputs['pi'], outputs['V']

        if self.cuda:
          probabilities = probabilities.cpu()
          value = value.cpu()
        probabilities = probabilities.detach().numpy()
        value = value.detach().numpy()
        if use_random_symmetry:
            probabilities = symmetries.invert_symmetries_pi(
                syms_used, probabilities)
        
        return probabilities, value.flatten()

    def bootstrap(working_dir):
        raise NotImplementedError


if __name__ == "__main__":
    input_ = np.zeros((1, 18, 19, 19)).astype(np.float32)
    input_[:, 16, :, :] = 1.
    # batch = torch.from_numpy(input_)

    fn = "/private/home/zhuoyuan/AlphaGo/ELF2_models/save-1661000.bin"
    # fn = "/Users/zhuoyuan/Exp/AlphaGo/ELF2_models/save-1661000.bin"
    model = DualNetwork(fn)

    # to run directly
    # res = model.model(batch)
    
    position = go.Position()
    prob, val = model.run(position)

    for i in range(19):
        print(prob[i*19:(i+1)*19])
    print(prob[-1])
    print(val)
