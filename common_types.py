from dataclasses import dataclass, field


@dataclass
class CheckpointInfo:
    epochs_elapsed: int = 0
    total_seconds_elapsed: int = 0
    train_loss: float | None = None
    train_loss_history: list[float] = field(default_factory=list)
    validation_loss: float | None = None

@dataclass
class TrainingSettings:
    dataset: str
    n_epochs: int = 10
    checkpoint_every_n_epochs: int = 5
    tensorboard_enabled: bool = False

@dataclass
class TAMHyperParameters:
    """
    n_nodes (int): The number of nodes/sensors.
    window_size (int): The number of timestamps in one sample.
    n_classes (int): The number of classes.
    n_gnn (int): The number of GNN modules.
    gsl_type (str): The type of GSL block.
    n_hidden (int): The number of hidden parameters in GCN layers.
    alpha (float): Saturation rate for GSL block.
    k (int): The maximum number of edges from one node.
    """
    n_nodes: int
    window_size: int
    n_classes: int
    n_gnn: int = 1
    gsl_type: str = 'relu'
    n_hidden: int = 1024
    alpha: float = 0.1
    k: int | None = None
    split_by: tuple[list[int]] | None = None
    k_additional: int | None = None
