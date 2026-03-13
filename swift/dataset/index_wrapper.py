from torch.utils.data import Dataset


class IndexDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset


    def __getitem__(self, idx):
        item = self.dataset[idx]

        if isinstance(item, dict):
            item = dict(item)
            item["sample_idx"] = idx
            return item

        if isinstance(item, tuple):
            return (*item, idx)

        return {"data": item, "sample_idx": idx}


    def __len__(self):
        return len(self.dataset)