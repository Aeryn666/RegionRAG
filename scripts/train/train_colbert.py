import os
from pathlib import Path

import configue
import typer

from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.gpu_stats import print_gpu_utilization

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(config_file: Path) -> None:
    """
    Training script for ColVision models.

    Args:
        config_file (Path): Path to the configuration file.
    """
    print_gpu_utilization()

    print("Loading config")
    config = configue.load(config_file, sub_path="config")

    # configure output_dir in config
    if config.output_dir is None:
        config.output_dir = os.path.join("work_dirs", config_file.stem)

    config.tr_args.output_dir = config.output_dir

    config.tr_args.logging_dir = os.path.join(
        config.output_dir, "runs", config.tr_args.logging_dir.split('/')[-1]
    )

    print("Creating Setup")
    if isinstance(config, ColModelTrainingConfig):
        training_app = ColModelTraining(config)
    else:
        raise ValueError("Config must be of type ColModelTrainingConfig")

    if config.run_train:
        print("Training model")
        training_app.train()
        training_app.save()
        os.system(f"cp {config_file} {training_app.config.output_dir}/training_config.yml")

    print("Done!")


if __name__ == "__main__":
    app()
