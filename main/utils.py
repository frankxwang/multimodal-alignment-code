import torch

# TODO: Fix this, this biases towards longer sequences since the mean is not masked
def score_logits_sequence(logits, labels, padding_id=-100, attn_mask=None):
    if attn_mask is not None:
        labels[attn_mask == 0] = padding_id
    scores = torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), labels, reduction="none", ignore_index=padding_id)
    return masked_mean(scores, labels != padding_id, dim=1)


# mask is 1 if we include it
def masked_mean(scores, mask, dim=-1):
    return torch.sum(scores * mask, dim=dim) / torch.sum(mask, dim=dim)


class RaggedList:
    def __init__(self, ragged_list):
        self.lengths = [len(l) for l in ragged_list]
        self.ragged_list = ragged_list

    def flatten(self):
        return [val for l in self.ragged_list for val in l]

    # if we have a list L mapping to each row, duplicate each element
    # based on the number of elements in each row, and flatten
    def flatten_broadcast(self, other_list):
        return [l for i, l in enumerate(other_list) for _ in range(self.lengths[i])]

    def unflatten(self, flat_list, reduce=None):
        out = []
        ind = 0
        for length in self.lengths:
            if reduce:
                out.append(reduce(flat_list[ind:ind+length]))
            else:
                out.append(flat_list[ind:ind+length])
            ind += length

        return out


# maybe make this function at some point
# def run_batch(func, batch_size, data, batch_args={"images", "text"}):
#     results = []
#     for i in range(0, len(data), batch_size):
#         results.append(func(data)
