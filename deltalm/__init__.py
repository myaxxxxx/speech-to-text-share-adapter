from fairseq.models import register_model_architecture
# from transformer_base import hytransformer

# from .transformer_legacy import base_architecture

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from fairseq.models.transformer.transformer_base import (
    TransformerModelBase,
)
from .transformer_base import PTransformerModelBase
from .options import *
# from .transformer_encoder import HyTransformerEncoderBase



from fairseq.models import MODEL_REGISTRY, ARCH_MODEL_INV_REGISTRY



from fairseq.models.transformer import (
    TransformerModel,
)



from .deltalm import DeltaLMEncoder
from .deltalm import DeltaLMDecoder

@register_model("deltalm")
class DeltaLMModel(TransformerModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--pretrained-deltalm-checkpoint",
            type=str,
            metavar="STR",
        )
        # add 
        parser.add_argument(
            "--prefix_length",
            type=int,
            help="# of channels in Conv1d (s2t_transformer) subsampling layers",
        )
        parser.add_argument(
            "--use_prefix_decoder",
            type=bool,
            help="# of channels in Conv1d (s2t_transformer) subsampling layers",
        )

        parser.add_argument(
            "--prefix_decoder_length",
            type=int,
            help="# of channels in Conv1d (s2t_transformer) subsampling layers",
        )

        parser.add_argument(
            "--frozen_plm",default='False', action='store_true',
            # action='store_true',
            # default=False,
            # type=bool,
            help="# of channels in Conv1d (s2t_transformer) subsampling layers",
        )



    @classmethod
    def build_encoder(cls, args, tgt_dict, embed_tokens):


        encoder =  DeltaLMEncoder(TransformerConfig.from_namespace(args), tgt_dict, embed_tokens)
        print(cls)
        print(args.frozen_plm)
        print(type(args.frozen_plm))
        if args.frozen_plm == "True":
            print(6666666)
            exit()
            for name, parameter in encoder.named_parameters():
                if 'prefix_encoder' in name:
                    parameter.requires_grad = True
                else:
                    parameter.requires_grad = False               
        print(encoder)
        total_num = sum(p.numel() for p in encoder.parameters())
        trainable_num = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print({'Total': total_num, 'Trainable': trainable_num})
        return encoder
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = DeltaLMDecoder(TransformerConfig.from_namespace(args), tgt_dict, embed_tokens)
        if args.frozen_plm == "True":
            exit()
            for name, parameter in decoder.named_parameters():
                    parameter.requires_grad = False
        return decoder

@register_model_architecture(
    "deltalm", "deltalm_base"
)
def base_architecture(args):
    args.encoder_embed_dim = 768
    args.encoder_ffn_embed_dim = 3072
    args.encoder_layers = 12
    args.encoder_attention_heads = 12
    args.encoder_normalize_before = False
    args.encoder_learned_pos = True
    args.decoder_embed_dim = 768
    args.decoder_ffn_embed_dim = 3072
    args.decoder_layers = 6
    args.decoder_attention_heads = 12
    args.decoder_normalize_before = False
    args.decoder_learned_pos = True
    args.activation_fn = "gelu"
    args.no_scale_embedding = True
    args.layernorm_embedding = True
    args.max_positions = 512


@register_model_architecture(
    "deltalm", "deltalm_large"
)
def large_architecture(args):

    base_architecture(args)
    args.encoder_embed_dim = 1024
    args.encoder_ffn_embed_dim = 4096
    args.encoder_layers = 24
    args.encoder_attention_heads = 16
    args.encoder_normalize_before = False
    args.decoder_embed_dim = 1024
    args.decoder_ffn_embed_dim = 4096
    args.decoder_layers = 12
    args.decoder_attention_heads = 16
    args.decoder_normalize_before = False
    args.layernorm_embedding = False