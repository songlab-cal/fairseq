from fairseq.models import register_model_architecture
from fairseq.models.roberta import RobertaModel, RobertaHubInterface
from fairseq.models.roberta.model import base_architecture
from fairseq.tasks.sentence_prediction import SentencePredictionTask
from fairseq.tasks.masked_lm import MaskedLMTask
import fairseq
import fairseq.utils
import torch
import pandas as pd
from dataclasses import dataclass
import os
import math
from fairseq import metrics, modules, utils
from fairseq.criterions import register_criterion, FairseqCriterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
import json


def seq2tokens(seq, dictionary, add_bos):
    assert ' ' not in seq, f'Expected regular sequence without space separators, found space in "{seq}"'
    sentence = ('<s> ' if add_bos else '') + seq2sentence(seq).replace('X', '<mask>') + ' </s>'
    return dictionary.encode_line(sentence, append_eos=False, add_if_not_exist=False).long()


def seq2sentence(seq):
    """Convert sequence to string of space-separated letters"""
    return ' '.join(list(seq))


@dataclass
class RoBERTa:
    interface: RobertaHubInterface

    @classmethod
    def load(cls, model_dir, checkpoint_name='checkpoint.pt') -> 'RoBERTa':
        full_model_dir = os.path.abspath(model_dir)
        if os.path.exists(full_model_dir):
            interface = RobertaModel.from_pretrained(
                full_model_dir,
                checkpoint_name,
                full_model_dir,
                bpe=None
            )
            interface.eval()
            return cls(interface=interface)
        else:
            raise FileNotFoundError(f'Model not found: {full_model_dir}')

    def predict_proba(self, seq):
        raise NotImplementedError()


@dataclass
class RoBERTaSeq2Seq(RoBERTa):

    def predict_proba(self, seq, remove_special=True, return_all_hiddens=False):
        """
        Get model output probabilities for all positions in a single sequence (slow, consider batching for large inputs)
        :param seq: Input sequence (str or Bio.Seq)
        :param remove_special: remove special dictionary tokens from predicted probability matrix
        :param return_all_hiddens: return attention from all layers (used with return_attention)
        :return: 2D dataframe of output token probabilities (columns) for each input position (rows)
        """
        add_bos = self._is_adding_bos()
        tokens = self._source_seq_tokens(seq, add_bos=add_bos)
        with torch.no_grad():
            logits, extra = self.interface.model(
                tokens.unsqueeze(0),
                features_only=False,
                return_all_hiddens=return_all_hiddens
            )
            pred = logits.softmax(dim=-1)[0]
            # remove EOS token
            pred = pred[:-1, :]
            if add_bos:
                # remove BOS token
                pred = pred[1:, :]
        pred = pd.DataFrame(pred.numpy(), columns=self.interface.task.target_dictionary.symbols)
        if remove_special:
            pred.drop(['<s>', '<pad>', '</s>', '<unk>', '<mask>'], axis=1, inplace=True)
        return pred

    def _is_adding_bos(self):
        if isinstance(self.interface.task, SentencePredictionTask):
            return True
        elif isinstance(self.interface.task, MaskedLMTask):
            return True
        return ValueError(f'Unknown task: {type(self.interface.task)}')

    def _source_seq_tokens(self, seq, add_bos):
        return seq2tokens(seq, dictionary=self.interface.task.source_dictionary, add_bos=add_bos)

    def _target_seq_tokens(self, seq, add_bos):
        return seq2tokens(seq, dictionary=self.interface.task.target_dictionary, add_bos=add_bos)


@register_model_architecture('roberta','roberta_small')
def roberta_small_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 256)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    base_architecture(args)

@register_model_architecture('roberta', 'antiBERTa')
def antiBERTa_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    base_architecture(args)


@register_criterion('smooth_masked_lm')
class SmoothMaskedLmLoss(FairseqCriterion):
    """
    Label-smoothed implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, task, label_smoothing):
        super().__init__(task)
        self.eps = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        masked_tokens = sample['target'].ne(self.padding_idx)

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if masked_tokens.device == torch.device('cpu'):
            if not masked_tokens.any():
                masked_tokens.fill_(True)
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        net_output = model(**sample['net_input'], masked_tokens=masked_tokens)
        logits, _ = net_output
        targets = model.get_targets(sample, [logits])
        targets = targets[masked_tokens]

        lprobs = model.get_normalized_probs(net_output, log_probs=True)

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs.view(-1, lprobs.size(-1)),
            targets.view(-1, 1),
            self.eps,
            reduce=reduce,
            ignore_index=self.padding_idx
        )

        sample_size = masked_tokens.int().sum()
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


SUBS_MAT = json.load(open('subs_mat.json', 'r'))

def aa_smoothed_nll_loss(lprobs, target, epsilon, token_dists, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    samp_dists = token_dists[target.flatten(), :].reshape(lprobs.shape)
    aa_loss = (-lprobs * samp_dists).sum(dim=-1, keepdim=True)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        aa_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        aa_loss = aa_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        aa_loss = aa_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * aa_loss
    return loss, nll_loss

@register_criterion('aa_masked_lm')
class AaMaskedLmLoss(FairseqCriterion):
    """
    Label-smoothed implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, task, label_smoothing):
        super().__init__(task)
        self.eps = label_smoothing

        source_dict = task.source_dictionary
        num_symbols = len(source_dict)
        self.token_dists = torch.zeros((num_symbols, num_symbols))
        for sym1 in SUBS_MAT:
            for sym2 in SUBS_MAT:
                self.token_dists[source_dict.index(sym1), source_dict.index(sym2)] = SUBS_MAT[sym1][sym2]

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        masked_tokens = sample['target'].ne(self.padding_idx)

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if masked_tokens.device == torch.device('cpu'):
            if not masked_tokens.any():
                masked_tokens.fill_(True)
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )

        net_output = model(**sample['net_input'], masked_tokens=masked_tokens)
        logits, _ = net_output
        targets = model.get_targets(sample, [logits])
        targets = targets[masked_tokens]

        lprobs = model.get_normalized_probs(net_output, log_probs=True)

        loss, nll_loss = aa_smoothed_nll_loss(
            lprobs.view(-1, lprobs.size(-1)),
            targets.view(-1, 1),
            self.eps,
            self.token_dists.to(lprobs.get_device()),
            reduce=reduce,
            ignore_index=self.padding_idx
        )

        sample_size = masked_tokens.int().sum()
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
