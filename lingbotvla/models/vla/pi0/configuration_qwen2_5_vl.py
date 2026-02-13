from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Qwen2_5_VLVisionConfig(PretrainedConfig):
    model_type = "qwen2_5_vl"
    def __init__(
        self,
        depth=32,
        hidden_size=1280,
        num_heads=16,
        in_channels=3,
        out_hidden_size=2048,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        fullatt_block_indexes=[7, 15, 23, 31],
        window_size=112,
        tokens_per_second=2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_hidden_size = out_hidden_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.window_size = window_size
        self.tokens_per_second = tokens_per_second
        self._attn_implementation = kwargs.get("attn_implementation", "eager")

class Qwen2_5_VLConfig(PretrainedConfig):
    model_type = "qwen2_5_vl"
    def __init__(
        self,
        vision_config=None,
        vocab_size=151936,
        hidden_size=2048,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=2,
        rms_norm_eps=1e-06,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_dropout=0.0,
        intermediate_size=11008,
        hidden_act="silu",
        use_cache=True,
        use_sliding_window=False,
        sliding_window=32768,
        max_window_layers=70,
        norm_qkv=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.use_cache = use_cache
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.norm_qkv = norm_qkv
        
        # 必须注入 mrope 需要的 section
        if rope_scaling is None:
            self.rope_scaling = {"type": "mrope", "mrope_section": [16, 24, 24]}
        else:
            self.rope_scaling = rope_scaling

        if isinstance(vision_config, dict):
            self.vision_config = Qwen2_5_VLVisionConfig(**vision_config)
        elif vision_config is None:
            self.vision_config = Qwen2_5_VLVisionConfig()
        else:
            self.vision_config = vision_config
            
        super().__init__(**kwargs)
        self._attn_implementation = kwargs.get("attn_implementation", "eager")