import d3rlpy
from typing import Optional, List, Sequence
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.torch.encoders import VectorEncoder, VectorEncoderWithAction


class CustomDenseEncoder(d3rlpy.models.encoders.EncoderFactory):
    TYPE = "custom-encoder"

    def __init__(
        self,
        activation: str = "relu",
        hidden_units: List[int] = [256, 256, 256, 256],
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
    ):
        self.hidden_units = hidden_units
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

    def create(self, observation_shape) -> VectorEncoder:
        factory = VectorEncoderFactory(
            hidden_units=self.hidden_units,
            activation=self.activation,
            use_dense=True,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
        )
        return factory.create(observation_shape)

    def create_with_action(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        discrete_action: bool = False,
    ) -> VectorEncoderWithAction:
        factory = VectorEncoderFactory(
            hidden_units=self.hidden_units,
            activation=self.activation,
            use_dense=True,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
        )
        return factory.create_with_action(
            observation_shape, action_size, discrete_action
        )

    def get_params(self, deep=False):
        return {"hidden_units": self.hidden_units,
                "activation": self.activation,
                "use_batch_norm": self.use_batch_norm,
                "dropout_rate": self.dropout_rate}
