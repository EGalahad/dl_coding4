from torch.utils.data import Dataset
import json
import os
import sys
import torch

from dictionary import Dictionary

from transformers import AutoTokenizer

# the model wants batch to be
# batch["input_ids"]: [batch_size, 4, [SEP] content [SEP] question [SEP] choice [SEP]]
# batch["labels"]: [batch_size]
# batch["token_type_ids"]: [batch_size, 4, ...]
# batch["attention_mask"]: [batch_size, 4, ...]

class CLSDataset(Dataset):

    def __init__(self,
                 max_len=512,
                 model_name = "bert-base-chinese",
                 data_path=os.path.join(os.path.dirname(__file__),
                                        "../Datasets/CLS/"),
                 split="train",
                 device="cpu"):

        self.filename = os.path.join(data_path, "{}.json".format(split))

        with open(self.filename, encoding="utf-8") as f:
            self.data = json.load(f)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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

        self.num_choices = 4
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

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
        for _ in range(self.num_choices - len(choices)):
            choices.append("")

        label = torch.tensor(label, dtype=torch.long)
        return {
            "content": content,
            "q": q,
            "choices": choices,
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
        contents = [sample["content"] for sample in samples]
        questions = [sample["q"] for sample in samples]
        choices = [sample["choices"] for sample in samples]

        batch_size = len(samples)
        num_choices = self.num_choices

        first_sentences = [[content] * num_choices for content in contents]
        second_sentences = [[question + '[SEP]' + choice for choice in choices[i]] for i, question in enumerate(questions)]

        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        # first_sentences, second_sentences: [batch_size * num_choices]

        batch = self.tokenizer(first_sentences,
                               second_sentences,
                               max_length=self.max_len,
                               truncation='only_first',
                               padding='max_length',
                               return_tensors="pt")
        # batch["input_ids"]: [batch_size * num_choices, max_len]
        # batch["attention_mask"]: [batch_size * num_choices, max_len]
        # batch["token_type_ids"]: [batch_size * num_choices, max_len]

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["targets"] = torch.tensor([sample["label"] for sample in samples], dtype=torch.long)
        # batch["targets"]: [batch_size]
        batch = {k: v.to(self.device) for k, v in batch.items()}
        return batch
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
