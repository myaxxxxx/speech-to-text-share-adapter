# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
from typing import Optional, Dict, List, Any, Tuple

from omegaconf import DictConfig
from argparse import Namespace

import torch
from torch import nn, Tensor
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerModel,
    base_architecture,
    transformer_wmt_en_de_big,
    transformer_iwslt_de_en,
    TransformerDecoder, TransformerEncoder)

logger = logging.getLogger(__name__)

class PrefixTransformer(TransformerModel):
    """
    See "The Power of Scale for Parameter-Efficient Prompt Tuning (Lester et al., 2021)"
    """

    @staticmethod
    def add_args(parser):
        # fmt: off
        super(PrefixTransformer, PrefixTransformer).add_args(parser)
        # Prompt tuning
        parser.add_argument('--prefix-init', type=str, metavar='N', default='from-vocab',
                            help='encoder prefix embedding init method [from-vocab, uniform]')
        parser.add_argument('--layer-prefix', action='store_true', default=False,
                            help='use new hidden layer prefix embeddings at each enc/dec layer')
        parser.add_argument('--reparametrize-prefix', action='store_true', default=False,
                            help='use a MLP re-parametrize the prefix matrix')
        parser.add_argument('--reparametrize-dim', type=int, default=512,
                            help='Re-parametrization dimension to be used with MLP')
        # fmt: on

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        self.args = args
        for n, p in self.named_parameters():
            if 'prefix_' not in n:
                p.requires_grad = False

    @classmethod
    def build_model(cls, args, task):
        # set any default arguments
        prefix_transformer(args)

        if args.layer_prefix and args.encoder_prefix_length != 0:
            encoder_prefixes = nn.ModuleList([PrefixLayers(args.encoder_prefix_length, args.encoder_embed_dim,
                                                           args.reparametrize_prefix, args.reparametrize_dim)
                                              for i in range(args.encoder_layers - 1)])
        else:
            encoder_prefixes = None

        if args.layer_prefix and args.decoder_prefix_length != 0:
            decoder_prefixes = nn.ModuleList([PrefixLayers(args.decoder_prefix_length, args.decoder_embed_dim,
                                                           args.reparametrize_prefix, args.reparametrize_dim)
                                              for i in range(args.decoder_layers - 1)])
        else:
            decoder_prefixes = None

        cls.encoder_prefixes = encoder_prefixes
        cls.decoder_prefixes = decoder_prefixes

        return super().build_model(args, task)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return PrefixTransformerEncoder(args, src_dict, embed_tokens, cls.encoder_prefixes)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return PrefixTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
            decoder_prefixes=cls.decoder_prefixes)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        prefix_length = args.encoder_prefix_length + args.decoder_prefix_length
        num_embeddings = len(dictionary) - prefix_length
        padding_idx = dictionary.pad()

        emb = EmbeddingsWithPrefixes(num_embeddings, embed_dim, padding_idx, args.encoder_prefix_length, args.decoder_prefix_length)
        return emb

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None):

        # Do not enforce the key match due to the prefix params
        strict = False

        if 'encoder.embed_tokens.prefix_embeddings.weight' in state_dict:
            return super().load_state_dict(state_dict, strict)

        token_embeddings = state_dict['encoder.embed_tokens.weight']

        # initialization of prefix tokens
        if self.args.prefix_init == 'from-vocab':
            sample_tokens_idx = random.sample(range(0, token_embeddings.shape[0]),
                                              self.encoder.embed_tokens.prefix_embeddings.weight.shape[0])
            prefix_dict = {'weight': token_embeddings[sample_tokens_idx]}
            info = 'encoder.embed_tokens.prefix_embeddings.weight is initialized from vocabulary'
        else:
            prefix_dict = {'weight': self.encoder.embed_tokens.prefix_embeddings.weight}
            info = 'encoder.embed_tokens.prefix_embeddings.weight is uniformly initialized'

        if 'decoder.output_projection.weight' not in state_dict:
            state_dict['decoder.embed_tokens.weight'] = torch.cat([token_embeddings,
                                                                        prefix_dict['weight'][1:, :]], 0)
            logger.info('decoder.output_projection.weight is updated by adding prefix embeddings')

        state = super().load_state_dict(state_dict, strict)

        logger.info(f'missing keys: {state.missing_keys}')
        logger.info(f'unexpected keys: {state.unexpected_keys}')

        self.encoder.embed_tokens.token_embeddings.load_state_dict({'weight': token_embeddings}, True)
        logger.info(f'encoder.embed_tokens.token_embeddings.weight is initialized with encoder.embed_tokens.weight')

        self.encoder.embed_tokens.prefix_embeddings.load_state_dict(prefix_dict, True)
        logger.info(info)

        # Zero-ing the padding idx
        nn.init.constant_(
            self.encoder.embed_tokens.token_embeddings.weight[self.encoder.embed_tokens.token_embeddings.padding_idx],
            0)
        nn.init.constant_(
            self.encoder.embed_tokens.prefix_embeddings.weight[self.encoder.embed_tokens.prefix_embeddings.padding_idx],
            0)

        return state


class PrefixLayers(nn.Module):

    def __init__(self, prefix_length, hidden_dim, reparametrize_prefix=False, reparametrize_dim=None):
        super(PrefixLayers, self).__init__()
        self.prefix_length = prefix_length
        self.hidden_dim = hidden_dim
        self.reparametrize_prefix = reparametrize_prefix
        self.reparametrize_dim = reparametrize_dim
        self.prefix_weight = nn.Parameter(torch.zeros(prefix_length, hidden_dim))
        self.prefix_weight.data.normal_(mean=0, std=self.hidden_dim ** -0.5)
        if reparametrize_prefix:
            self.prefix_transform = nn.Sequential(
                nn.Linear(self.hidden_dim, self.reparametrize_dim),
                nn.Tanh(),
                nn.Linear(self.reparametrize_dim, self.hidden_dim))

    def get_prefix_length(self):
        return self.prefix_length

    def forward(self, bsz):
        if self.reparametrize_prefix:
            prefix_out = self.prefix_transform(self.prefix_weight)
        else:
            prefix_out = self.prefix_weight
        return prefix_out.unsqueeze(0).repeat(bsz,1,1)


class EmbeddingsWithPrefixes(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, padding_idx, encoder_prefix_length, decoder_prefix_length):
        super(EmbeddingsWithPrefixes, self).__init__()
        self.encoder_prefix_length = encoder_prefix_length
        self.decoder_prefix_length = decoder_prefix_length
        self.prefix_length = encoder_prefix_length + decoder_prefix_length
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.prefix_padding_idx = 0
        self.token_embeddings = nn.Embedding(self.num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.prefix_embeddings = nn.Embedding(self.prefix_length+1, embedding_dim, padding_idx=self.prefix_padding_idx)
        self.init_weights()

    def init_weights(self, range=None):
        nn.init.normal_(self.token_embeddings.weight, mean=0, std=self.embedding_dim ** -0.5)
        nn.init.constant_(self.token_embeddings.weight[self.padding_idx], 0)
        nn.init.normal_(self.prefix_embeddings.weight, mean=0, std=self.embedding_dim ** -0.5)
        nn.init.constant_(self.prefix_embeddings.weight[self.prefix_padding_idx], 0)

    def get_prefix_length(self):
        return self.prefix_length

    def forward(self, input):
        token_input = input.detach().clone()
        prefix_input = input.detach().clone()
        prefix_mask = input > self.num_embeddings - 1
        prefix_input[~prefix_mask] = self.prefix_padding_idx
        prefix_input[prefix_mask] = prefix_input[prefix_mask] - self.num_embeddings + 1
        token_input[prefix_mask] = self.padding_idx
        prefix_embs = self.prefix_embeddings(prefix_input)
        token_embs = self.token_embeddings(token_input)

        return prefix_embs + token_embs


class PrefixTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens, encoder_prefixes=None):
        super().__init__(args, dictionary, embed_tokens)

        self.prefixes = encoder_prefixes

    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for i, layer in enumerate(self.layers):
            if i > 0 and self.prefixes:
                bsz = src_tokens.shape[0]
                prefix = self.prefixes[i -1](bsz).transpose(0, 1)
                x[:self.prefixes[0].get_prefix_length(), :, :] = prefix

            x = layer(x, encoder_padding_mask=encoder_padding_mask if has_pads else None)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }


class PrefixTransformerDecoder(TransformerDecoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
        decoder_prefixes=None
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)
        self.prefixes = decoder_prefixes

    def get_output_projection_weight(self):
        tokens_embs = self.embed_tokens.token_embeddings.weight
        prefix_embs = self.embed_tokens.prefix_embeddings.weight
        return nn.Parameter(torch.cat([tokens_embs, prefix_embs[1:,:]], 0))

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        incremental_step: int = None,
    ):
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            if idx > 0 and self.prefixes:
                bsz = prev_output_tokens.shape[0]
                prefix = self.prefixes[idx - 1](bsz).transpose(0, 1)
                if incremental_step is not None:
                    if incremental_step < self.prefixes[0].get_prefix_length():
                        x[0, :, :] = prefix[incremental_step, :, :]
                else:
                    x[:self.prefixes[0].get_prefix_length(), :, :] = prefix

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)))
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}


# @register_model_architecture("prefix_transformer", "prefix_transformer")
# def prefix_transformer(args):
#     args.encoder_prefix_length = getattr(args, "encoder_prefix_length", 20)
#     args.decoder_prefix_length = getattr(args, "decoder_prefix_length", 20)
#     args.ignore_prefix_size = args.decoder_prefix_length
#     args.prefix_init = getattr(args, "prefix_init", "from-vocab")
#     args.reparametrize_prefix = getattr(args, "reparametrize_prefix", False)
#     args.reparametrize_dim = getattr(args, "reparametrize_dim", 512)
#     base_architecture(args)


# @register_model_architecture("prefix_transformer", "prefix_transformer_wmt_en_de_big")
# def prefix_transformer_wmt_en_de_big(args):
#     args.encoder_prefix_length = getattr(args, "encoder_prefix_length", 20)
#     args.decoder_prefix_length = getattr(args, "decoder_prefix_length", 20)
#     args.prefix_init = getattr(args, "prefix_init", "from-vocab")
#     args.reparametrize_prefix = getattr(args, "reparametrize_prefix", False)
#     args.reparametrize_dim = getattr(args, "reparametrize_dim", 512)
#     transformer_wmt_en_de_big(args)


# @register_model_architecture("prefix_transformer", "prefix_transformer_iwslt_de_en")
# def prefix_transformer_iwslt_de_en(args):
#     args.encoder_prefix_length = getattr(args, "encoder_prefix_length", 20)
#     args.decoder_prefix_length = getattr(args, "decoder_prefix_length", 20)
#     args.prefix_init = getattr(args, "prefix_init", "from-vocab")
#     args.reparametrize_prefix = getattr(args, "reparametrize_prefix", False)
#     args.reparametrize_dim = getattr(args, "reparametrize_dim", 512)
#     transformer_iwslt_de_en(args)


# @register_model_architecture("prefix_transformer", "prefix_transformer_mbart_large")
# def prefix_transformer_mbart_large(args):
#     args.encoder_prefix_length = getattr(args, "encoder_prefix_length", 20)
#     args.decoder_prefix_length = getattr(args, "decoder_prefix_length", 20)
#     args.prefix_init = getattr(args, "prefix_init", "from-vocab")
#     args.reparametrize_prefix = getattr(args, "reparametrize_prefix", False)
#     args.reparametrize_dim = getattr(args, "reparametrize_dim", 512)
#     transformer_mbart_large(args)