from model import *
from dataset import *

import numpy as np
import torch
import torch.optim as opt

import argparse

def set_seed(seed):
    """Set the various seeds and flags to ensure deterministic performance

        Args:
            seed: The random seed
    """
    torch.backends.cudnn.deterministic = True   # Note, can impede performance
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model_only(filename, num_classes, device):
    """Load a model

        Args:
            filename: The name of the file with the pretrained model parameters
            num_classes: The number of classes available to predict
            weights: The weights to apply to the classes
            device: The device on which to run

        Returns:
            A tuple composed (in order) of the model, loss function, and optimiser
    """
    model = UNet(1, n_classes = num_classes, depth = 4, n_filters = 16, y_range = (0, num_classes - 1))
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    return model

def process_view(input_dir, vertex_pass, view, num_classes):
    torch.set_default_tensor_type(torch.FloatTensor)

    the_seed = 42
    batch_size = 32
    image_path = f"{input_dir}/Images_{view}/"

    device = torch.device('cpu')

    filename = f"{input_dir}/outputs/models/pass{vertex_pass}/{view}/uboone_hd_accel_19.pt"
    #output_filename = f"PandoraUnet_DLVertexing_UBoone_{vertex_pass}_{view}.pt"
    output_filename = f"pytorch_1_0_1_uboone_hd_accel_19.pt"

    set_seed(the_seed)

    print("Getting data...")
    bunch = SegmentationBunch(image_path, "Hits", "Truth", batch_size=batch_size, valid_pct = 0.15, device=device)

    print("Loading model...")
    model = load_model_only(filename, num_classes, device)

    input_examples = ()

    print("Tracing...")
    for batch in bunch.train_dl:
        x, _ = batch
        input_examples = (x.cpu())
        break

    sm = torch.jit.trace(model, input_examples)
    sm.save(output_filename)
    print(f"Done for view {view}!")

print(f"Starting...")
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--vertex_pass", type=int, required=True)
parser.add_argument("--n_classes", type=int, required=True)
args = parser.parse_args()

print(f"Running on {args.input_dir}...")
print(f"Which is pass {args.vertex_pass}...")
print(f"With {args.n_classes} classes...")

for view in ["U", "V", "W"]:
    print(f"Processing view {view}...")
    process_view(args.input_dir, args.vertex_pass, view, args.n_classes)
