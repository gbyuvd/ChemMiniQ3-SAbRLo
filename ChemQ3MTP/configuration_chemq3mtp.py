# configuration_chemq3mtp.py
from transformers import Qwen2Config

class ChemQ3MTPConfig(Qwen2Config):
    """
    Configuration class for ChemQ3MTP model.
    """
    model_type = "chemq3_mtp"
    
    def __init__(
        self,
        num_future_tokens: int = 3,
        horizon_weights = None,
        use_mtp_training: bool = True,
        entropy_controller_config = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_future_tokens = num_future_tokens
        self.horizon_weights = horizon_weights or [0.9 ** i for i in range(num_future_tokens)]
        self.use_mtp_training = use_mtp_training
        self.entropy_controller_config = entropy_controller_config or {
            "min_entropy": 0.5,
            "max_entropy": 3.0,
            "target_entropy": 1.5,
            "adaptation_rate": 0.01
        }