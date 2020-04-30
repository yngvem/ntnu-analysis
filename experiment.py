"""
Example of running a single experiment of unet in the head and neck data.
The json config of the main model is 'examples/json/unet-sample-config.json'
All experiment outputs are stored in '../../hn_perf/logs'.
After running 3 epochs, the performance of the training process can be accessed
as log file and perforamance plot.
In addition, we can peek the result of 42 first images from prediction set.
"""

import argparse
from pathlib import Path

import tensorflow as tf
from comet_ml import Experiment as CometEx
from deoxys.customize import custom_loss
from deoxys.experiment import Experiment
from deoxys.keras import backend as K
from deoxys.model.losses import Loss, loss_from_config
from deoxys.utils import is_keras_standalone, read_file


@custom_loss
class BinaryFbetaLossNoSquare(Loss):
    r"""A generalisation of the f_\beta loss that doesn't square the probabilities.
    """

    def __init__(self, reduction="auto", name="binary_fbeta", beta=1):
        if is_keras_standalone():
            # use Keras default reduction
            super().__init__("sum_over_batch_size", name)
        else:
            super().__init__(reduction, name)

        self.beta = beta

    def call(self, target, prediction):
        size = len(prediction.get_shape().as_list())
        reduce_ax = list(range(1, size))
        eps = 1e-8

        true_positive = K.sum(prediction * target, axis=reduce_ax)
        target_positive = K.sum((target), axis=reduce_ax)
        predicted_positive = K.sum((prediction), axis=reduce_ax)

        fb_numerator = (1 + self.beta ** 2) * true_positive + eps
        fb_denominator = (self.beta ** 2) * target_positive + predicted_positive + eps

        return 1 - fb_numerator / fb_denominator


@custom_loss
class FusedLoss(Loss):
    """Used to sum two or more loss functions.
    """

    def __init__(
        self, loss_configs, loss_weights=None, reduction="auto", name="fused_loss"
    ):
        if is_keras_standalone():
            # use Keras default reduction
            super().__init__("sum_over_batch_size", name)
        else:
            super().__init__(reduction, name)
        self.losses = [loss_from_config(loss_config) for loss_config in loss_configs]

        if loss_weights is None:
            loss_weights = [1] * len(self.losses)
        self.loss_weights = loss_weights

    def call(self, target, prediction):
        loss = None
        for loss_class, loss_weight in zip(self.losses, self.loss_weights):
            if loss is None:
                loss = loss_weight * loss_class(target, prediction)
            else:
                loss += loss_weight * loss_class(target, prediction)

        return loss


if __name__ == "__main__":
    if not tf.test.is_gpu_available():
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("log_folder")
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--model_checkpoint_period", default=25, type=int)
    parser.add_argument("--prediction_checkpoint_period", default=25, type=int)
    parser.add_argument("--comet_keyfile", default="comet_key.txt", type=str)
    args = parser.parse_args()

    # Create comet logger
    key_file = Path(args.comet_keyfile)
    if key_file.is_file():
        with key_file.open("r") as f:
            api_key = f.read().strip()

        experiment = CometEx(
            api_key=api_key, project_name="ntnu-mri", workspace="yngvem",
        )
        experiment.add_tag(args.config_file.split("/")[-1])

    # Run deep learning experiment
    config = read_file(args.config_file)
    (
        Experiment(log_base_path=args.log_folder)
        .from_full_config(config)
        .run_experiment(
            train_history_log=True,
            model_checkpoint_period=args.model_checkpoint_period,
            prediction_checkpoint_period=args.prediction_checkpoint_period,
            epochs=args.epochs,
        )
        .plot_performance()
        .plot_prediction(masked_images=[i for i in range(42)])
    )
