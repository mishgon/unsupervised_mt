import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits


def translation_loss(translated, target):
    """
    translated: length x batch_size x vocab_size
    target: length x batch_size
    """
    assert translated.size(0) == target.size(0)

    criterion = nn.NLLLoss()
    loss = 0
    for t in range(translated.size(0)):
        loss += criterion(translated[t], target[t])

    return loss


def classification_loss(logits, language):
    """
    logits: batch_size x 1
    """
    return binary_cross_entropy_with_logits(logits, torch.full(logits.shape, language == 'tgt', device=logits.device))
