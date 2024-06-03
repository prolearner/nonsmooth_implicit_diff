import torch 

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False, device="cpu"):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        self.device = device
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = [t.to(device) for t in tensors]

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len).to(self.device)
        else:
            self.indices = torch.tensor(list(range(self.dataset_len))).to(self.device)
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration

        indices = self.indices[self.i : self.i + self.batch_size]

        if self.tensors[0].is_sparse:
            # still slow
            batch = tuple(
                torch.cat([t[i].unsqueeze(0) for i in indices]) for t in self.tensors
            )
            # batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)

        else:
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)

        self.i += self.batch_size
        return batch