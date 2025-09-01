import argparse
import dataclasses
import os
from dataclasses import dataclass

import yaml
import sys
sys.path.append('./IINet')

@dataclass
class Options():
    """"Dataclass for housing experiment flags."""

    seed: int = 0

    ################################### logs ###################################
    # log directory for training.
    logdir: str = "./run"
    # experiment name
    expname: str = "exp"

    # frequency for summary
    summary_freq: int = 40
    summary_freq_eval: int = 10
    # frequency for saving model
    save_freq: int = 4
    # frequency for eval model
    eval_freq: int = 1
    # if save eval imgs
    save_eval: bool = False

    ################################### data ###################################
    # which dataset should we use?
    dataset: str = "sceneflow"
    
    # base dataset path.
    dataset_path: str = "./20250701_144157_images/20250701_144230_445"
    ratio: float = 0.9

    # number of dataloader workers to use.
    num_workers: int = 8

    # the split to use, script dependant.
    split: str = "train"

    # image size input to the network. Used in dataloaders and projectors.
    train_width: int = 512
    train_height: int = 384

    # image size input to validate.
    val_width: int = 960
    val_height: int = 512

    ############################## hyperparameters #############################
    # learning rate
    lr: float = 1e-4
    # weight decay
    wd: float = 1e-4

    # number of sanity validation steps before training
    num_sanity_val_steps: int = 0

    # max number of iterations for training
    epochs: int = 90
    # for resuming
    start_epoch: int = 0

    # rescaling groundtruth
    disp_scale: int = 16

    # batch size
    batch_size: int = 16
    # validation batch size during training
    val_batch_size: int = 1

    # number of GPUs to use for training.
    gpus: int = 1

    # precision to use for training.
    precision: int = 16

    lrepochs: str = "10,15,18,21:2"


    ################################## models ##################################
    # mode for running [train, eval, test]
    run_mode: str = "train"

    #if finetuning
    ft: bool = False

    # loads model weights
    load_weights_from_checkpoint: str = "./checkpoints/sceneflow.tar"

    # modelpath to load pretrained model
    load_pretrained_path: str = './models'

    # pretrained weight name for backbones
    pre_weight_name: str = 'models/mobilenetv3_large_100_ra-f55367f5.pth'

    # image prior encoder
    image_encoder_name: str = "efficientnet"

    # final depth decoder.
    depth_decoder_name: str = "unet_pp"

    # matching encoder. resnet or fpn
    matching_encoder_type: str = "resnet"
    # number of channels for matching features
    matching_feature_dims: int = 16

    # scale to match features at. 1 a
    # quarter of image resolution.
    matching_scale: int = 2
    # scales to build multi score volume
    multiscale: int = 1
    # scale num to output depth
    out_scale: int = 4

    # max_disp used
    max_disp: int = 512

    # type of cost volume encoder.
    cv_encoder_type: str = "multi_scale_encoder"

    # type of cost volume to use. SimpleRecon's metadata model uses the
    # 'mlp_feature_volume' model. Also available in this repo is a simple dot
    # reduction model 'simple_cost_volume'
    feature_volume_type: str = "mlp_feature_volume"
    # style of cost aggregation
    dot_dim: int = 1

    # weights for different losses with multi-resolution, index 0 is the max resolution
    loss_weights: dict = dataclasses.field(default_factory=lambda: {"logRel": [0.25, 0.5, 0.75, 1.0],
                                                                    "grad": [1.0, 0.75, 0.5, 0.25],
                                                                    "normal": 0.5})



class OptionsHandler():
    """A class for handling experiment options.

        This class handles options files and optional CLI arguments for
        experimentation.
        
        The intended use looks like this:

            optionsHandler = options.OptionsHandler()
            # uses a config filename from args or populates flags from CLI
            optionsHandler.parse_and_merge_options()

            # optionally print
            optionsHandler.pretty_print_options()

        You could also load from a config file you choose and ignore one that 
        may be supplied in args.
            optionsHandler.parse_and_merge_options(config_filepath =
                                    os.path.join("configs", "test_config.yaml"))

        Options will be populated by an optional supplied config files first, 
        then overwritten by any changes provided in command line args. If a 
        required attribute is not defined in either, then an Exception is thrown.
        I want to add a new arg! What should I do? Well, easy. Add an entry in 
        the Options class and specify a type and default value. If this needs to 
        be a required arg, set None for a default value and also add its name
        as a string to the required_flags list in the OptionsHandler class's 
        initializer.

        There are two config files allowed. --config_file, then 
        --data_config_file. Order of overriding (last overrides above): 
            - config_file
            - data_config_file
            - CLI arguments

    """
    def __init__(self, required_flags=[]):
        """Sets up the class and stores required flags."""
        if required_flags is None:
            required_flags = []

        self.options = Options()
        self.required_flags = required_flags

        self.parser = argparse.ArgumentParser(description="SimpleRecon Options")
        self.parser.add_argument('--config_file', type=str, default="IINet/configs/models/dot_model.yaml")
        self.parser.add_argument('--data_config_file', type=str, default="IINet/configs/data/sceneflow.yaml")

        self.populate_argparse()

    def parse_and_merge_options(self, config_filepaths=None, ignore_cl_args=False):
        """Parses flags from a config file and CL arguments.
            Args:
                config_filepaths: str filepath to a .yaml or list of filepaths 
                to config files
                ignore_cl_args: optionally ignore CLI
                altogether, useful for debugging with a hardcoded config 
                filepath and in python notebooks.

            Raises:
                Exception: raised when required arguments aren't satisfied.
        """
        # parse args
        if not ignore_cl_args:
            cl_args = self.parser.parse_args()

        # load config file
        if config_filepaths is not None:
            # use config_filepath(s) provided here if available
            if isinstance(config_filepaths, list):
                for config_filepath in config_filepaths:
                    config_options = OptionsHandler.load_options_from_yaml(
                                                                config_filepath)
                    self.merge_config_options(config_options)
            else:
                config_options = OptionsHandler.load_options_from_yaml(
                                                               config_filepaths)
                self.merge_config_options(config_options)

            self.config_filepaths = config_filepaths

        elif (not ignore_cl_args and 
            (cl_args.config_file is not None or 
                cl_args.data_config_file is not None)):
            # if args tells us we should load from a file, then let's do that.
            self.config_filepaths = []

            # add from standard config first
            if cl_args.config_file is not None:
                config_options = OptionsHandler.load_options_from_yaml(
                                                            cl_args.config_file)
                self.merge_config_options(config_options)
                self.config_filepaths.append(cl_args.config_file)

            # then merge from a data config
            if cl_args.data_config_file is not None:
                config_options = OptionsHandler.load_options_from_yaml(
                                                    cl_args.data_config_file)
                self.merge_config_options(config_options)
                self.config_filepaths.append(cl_args.data_config_file)
        else:
            # no config has been supplied. Let's hope that we have required
            # arguments through command line.
            print("Not reading from a config_file.")
            config_options = None
            self.config_filepaths = None

        if not ignore_cl_args:
            # merge args second and overwrite everything that's come before
            self.merge_cl_args(cl_args)

        # now check that all required arguments are satisfied
        self.check_required_items()

    def populate_argparse(self):
        """Populates argparse arguments using Options attributes."""

        for field_name in self.options.__dataclass_fields__.keys():
            field_info = self.options.__dataclass_fields__[field_name]
            if field_info.type == bool:
                self.parser.add_argument(f'--{field_name}', action="store_true")
            else:
                self.parser.add_argument(
                                        f'--{field_name}', 
                                        type=field_info.type, 
                                        default=None,
                                    )

    def check_required_items(self):
        """Raises a flag if options isn't defined."""
        for required_flag in self.required_flags:
            if self.options.__getattribute__(required_flag) is None:
                raise Exception(f"Error! Missing required config argument '{required_flag}'")

    def merge_config_options(self, config_options):
        """"""

        # loop over loaded config and update those in self.options.
        for field_name in config_options.__dict__.keys():
            value = config_options.__getattribute__(field_name)
            self.options.__setattr__(field_name, value)

    def merge_cl_args(self, cl_args):
        # loop over loaded args and update those in self.options.
        for arg_pair in cl_args._get_kwargs():
            # this should be the only argument that doesn't match here.
            if arg_pair[0] == "config_file":
                continue

            if arg_pair[1] is not None:
                # check if type bool and if false, in that case ignore
                if isinstance(arg_pair[1], bool) and not arg_pair[1]:
                    continue

                if arg_pair[0] == "prediction_mlp_channels":
                    array = "".join(arg_pair[1]).split("_")
                    array = [int(dim) for dim in array]
                    self.options.__setattr__(arg_pair[0], array)
                else:
                    self.options.__setattr__(arg_pair[0], arg_pair[1])

    def pretty_print_options(self):
        print("########################### Options ###########################")
        print("")
        for field_name in self.options.__dataclass_fields__.keys():
            print("    ", field_name + ":", self.options.__getattribute__(field_name))
        print("")
        print("###############################################################")

    @staticmethod
    def load_options_from_yaml(config_filepath):
        stream = open(config_filepath, 'r')
        return yaml.load(stream, Loader=yaml.Loader)

    @staticmethod
    def save_options_as_yaml(config_filepath, options):
        with open(config_filepath, 'w') as outfile:
            yaml.dump(options, outfile, default_flow_style=False)


def handle_backwards_compat(opts):
    # modify older experiment configs if needed
    return opts
