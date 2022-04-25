# This is taken from https://github.com/EleutherAI/lm_perplexity. We will align to this method of producing model inputs
# so we can compare to Pile evaluation numbers.
def get_rolling_token_windows(token_list, prefix_token, max_seq_len, context_len):
    """
    - context_len allows for a rolling window context, allowing each prediction window to potentially
      condition on some context
    :param token_list: list
        List of tokens to be PREDICTED
    :param max_seq_len: int
        max_seq_len of model (or max_seq_len we want to use)
    :param context_len: int
        Amount of desired token context for prediction. Needs to be at least 1.
    :param prefix_token: token
        Dummy token like <eos> so the first token has something to condition on
    :return: generator
        Generator of tuples
            (input_tokens, pred_tokens)
        Note: Score only the last len(pred_tokens) logits of the LM
    """
    assert 1 <= context_len <= max_seq_len
    if not token_list:
        return
    # +1 offset, going from input->preds
    pred_len = max_seq_len - context_len + 1
    predicted = 0

    # Special handling for first window: predict all tokens
    first_seq_len = min(max_seq_len, len(token_list))
    yield (
        [prefix_token] + token_list[:first_seq_len - 1],
        token_list[:first_seq_len]
    )
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len
        yield (
            token_list[window_end - max_seq_len - 1:window_end - 1],
            token_list[window_end - window_pred_len:window_end],
        )
        predicted += window_pred_len


def compute_seq_and_weight(seq, y_start, max_seq_len, pad_id):
    y_len = len(seq) - y_start
    seq_len = len(seq)
    weight = [0] * y_start + [1] * y_len

    adj_max_seq_len = max_seq_len + 1
    seq = seq + [pad_id] * (adj_max_seq_len - seq_len)
    weight += [0] * (adj_max_seq_len - seq_len)

    return seq, weight