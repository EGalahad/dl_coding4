from torch.utils.data import Dataset
import json
import os
import sys
import torch

from dictionary import Dictionary

class CLSDataset(Dataset):

    def __init__(self,
                 data_path=os.path.join(os.path.dirname(cur_dir),
                                        "Datasets/CLS/"),
                 vocab_file="./vocab.txt",
                 split="train",
                 device="cpu"):

        self.filename = os.path.join(data_path, "{}.json".format(split))

        with open(self.filename, encoding="utf-8") as f:
            self.data = json.load(f)

        self.padding_idx = None
        self.cls_idx = self.bos_idx = None
        self.sep_idx = self.eos_idx = None

        self.vocab_file = vocab_file
        try:
            self.dictionary = Dictionary(extra_special_symbols=["<q>", "<cls>"])
            self.dictionary.add_from_file(self.vocab_file)
        except:
            self.dictionary = Dictionary(extra_special_symbols=["<q>", "<cls>"])
            self._init_vocab()
        self.vocab_size = len(self.dictionary)

        self.cls_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        self.pairs = []
        for article in self.data:
            content = article["Content"]
            for question in article['Questions']:
                q = question['Question']
                choices = question['Choices']
                label = self.cls_map[question['Answer']]
                self.pairs.append([content, q, choices, label])
        self.device = device
        self.split = split

    def __len__(self):
        return len(self.pairs)

    def _init_vocab(self):
        from tqdm import tqdm
        for article in tqdm(self.data, desc="Initializing vocab"):
            content = article["Content"] # one long string
            all_words = content
            for question in article['Questions']:
                q = question['Question']
                choices = question['Choices']
                all_words += q + "".join(choices)
            for word in all_words:
                if word == ["<", "q", "c", "l", "s", ">"]:
                    print(word, "escaped")
                    continue
                if word in ["A", "B", "C", "D"]:
                    continue
                self.dictionary.add_symbol(word)
        self.dictionary.save(self.vocab_file)

    # @profile
    def __getitem__(self, index):
        """
        Get a data pair from the dataset by index.

        This method is used by PyTorch DataLoader to retrieve individual data pairs
        from the dataset. It should be implemented to return the data pair at the
        specified index.

        Args:
            index (int): Index of the data pair.

        Returns:
            your_data_pair(obj): Data pair at the specified index.
        """
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        content, q, choices, label = self.pairs[index]
        content_id = self.dictionary.encode_line(content, add_if_not_exist=False, append_eos=False)
        q_id = self.dictionary.encode_line(q, add_if_not_exist=False, append_eos=False)
        choices_id = [self.dictionary.encode_line(choice, add_if_not_exist=False, append_eos=False) for choice in choices]
        label = torch.tensor(label, dtype=torch.long)
        return {
            "content": content_id,
            "q": q_id,
            "choices": choices_id,
            "label": label
        }
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

    # @profile
    def collate_fn(self, samples):
        """
        Collate function for DataLoader.

        This method is used by PyTorch DataLoader to process and batch the samples
        returned by the __getitem__ method. It should be implemented to return a
        batch of data in the desired format.

        Args:
            samples (list): List of data pairs retrieved using the __getitem__ method.

        Returns:
            your_batch_data: Batch of data in your desired format.
        """
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        max_content_len = max([len(sample["content"]) for sample in samples])
        max_q_len = max([len(sample["q"]) for sample in samples])
        max_choices_len = max([max([len(choice) for choice in sample["choices"]]) for sample in samples])
        
        batch_size = len(samples)
        
        contents = torch.empty(batch_size, max_content_len, dtype=torch.long, device=self.device)
        contents.fill_(self.dictionary.pad())
        
        qs = torch.empty(batch_size, max_q_len, dtype=torch.long, device=self.device)
        qs.fill_(self.dictionary.pad())
        
        choices = torch.empty(batch_size, 4, max_choices_len, dtype=torch.long, device=self.device)
        choices.fill_(self.dictionary.pad())

        for batch_idx, sample in enumerate(samples):
            contents[batch_idx, :len(sample["content"])].copy_(sample["content"])
            qs[batch_idx, :len(sample["q"])].copy_(sample["q"])
            for j, choice in enumerate(sample["choices"]):
                choices[batch_idx, j, :len(choice)].copy_(choice)
                
        labels = torch.stack([sample["label"] for sample in samples]).to(self.device)
        return {
            "content": contents,
            "q": qs,
            "choices": choices,
            "targets": labels
        }
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
