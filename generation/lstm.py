import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        self.padding_idx = dictionary.pad()
        self.dictionary = dictionary


class LMModel(BaseModel):

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        # Hint: Use len(dictionary) in __init__
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        self.vocab_size = len(dictionary)
        self.hidden_size = hidden_size = args.hidden_size
        self.num_layers = num_layers = args.num_layers
        self.embedding_dim = embedding_dim = args.embedding_dim

        self.embedding = nn.Embedding(len(dictionary), embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)

        self.fc_head = nn.Linear(hidden_size, len(dictionary))
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

    def logits(self, source, **unused):
        """
        Compute the logits for the given source.

        Args:
            source: The input data.
            **unused: Additional unused arguments.

        Returns:
            logits: The computed logits.
        """
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        source_embedded = self.embedding(source)
        output, _ = self.lstm(source_embedded)
        logits = self.fc_head(output)
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
        return logits

    def get_loss(self, source, target, reduce=True, **unused):
        logits = self.logits(source)
        lprobs = F.log_softmax(logits, dim=-1).view(-1, logits.size(-1))
        return F.nll_loss(
            lprobs,
            target.view(-1),
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

    @torch.no_grad()
    def generate(self, prefix, max_len=100, beam_size=None):
        """
        Generate text using the trained language model with beam search.

        Args:
            prefix (str): The initial words, like "白".
            max_len (int, optional): The maximum length of the generated text.
                                     Defaults to 100.
            beam_size (int, optional): The beam size for beam search. Defaults to None.

        Returns:
            outputs (str): The generated text.(e.g. "白日依山尽，黄河入海流，欲穷千里目，更上一层楼。")
        """
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        self.eval()

        if prefix == None:
            prefix = ""
        
        encoded_prefix = self.dictionary.encode_line(prefix, append_eos=False)
        encoded_prefix = torch.cat([torch.tensor([self.dictionary.bos()]), encoded_prefix])
        encoded_prefix = encoded_prefix.to(next(self.parameters()).device).unsqueeze(0)
        # add batch dimension [batch_size=1, seq_len]

        embedded_prefix = self.embedding(encoded_prefix) # [batch_size=1, seq_len, embedding_dim]

        output, (h, c) = self.lstm(embedded_prefix) # output: [batch_size=1, seq_len, hidden_size]
        # h, c: [num_layers, batch_size=1, hidden_size]

        # list of candidate word indices
        beam_list = [[word_idx.item() for word_idx in encoded_prefix[0]]] # beam_list: [beam_size, seq_len]
        # list of hidden states
        hidden_list = [(h, c)] # hidden_list: [beam_size, (h, c)]
        # list of log probabilities
        log_prob_list = [0] # log_prob_list: [beam_size]
        # list of output
        output_list = [output[0, -1, :]] # output_list: [beam_size, hidden_size]

        done_list = []
        done_log_prob_list = []

        for idx in range(max_len - len(prefix)):
            temp_beam_list = []
            temp_hidden_list = []
            temp_log_prob_list = []
            temp_output_list = []

            if len(beam_list) == 0:
                break

            for beam_idx, beam in enumerate(beam_list):
                current_hidden = hidden_list[beam_idx]
                current_output = output_list[beam_idx]

                logits = self.fc_head(current_output) # logits: [vocab_size]
                log_probs = F.log_softmax(logits, dim=-1) # log_probs: [vocab_size]

                log_probs_topk, word_indices_topk = torch.topk(log_probs, beam_size, dim=-1) 
                # log_probs_topk: [beam_size], word_indices_topk: [beam_size]

                for word_idx, log_prob in zip(word_indices_topk, log_probs_topk):
                    if word_idx.item() == self.dictionary.eos():
                        done_list.append(beam + [word_idx.item()])
                        done_log_prob_list.append(log_prob.item() + log_prob_list[beam_idx])
                        continue
                    
                    embedded_word = self.embedding(word_idx.unsqueeze(0).to(next(self.parameters()).device)) # embedded_word: [1, embedding_dim]
                    next_output, (next_h, next_c) = self.lstm(embedded_word, current_hidden) # next_output: [1, 1, hidden_size]

                    temp_beam_list.append(beam + [word_idx.item()])
                    temp_hidden_list.append((next_h, next_c))
                    temp_log_prob_list.append(log_prob.item() + log_prob_list[beam_idx])
                    temp_output_list.append(next_output[0, -1, :])
            
            if len(temp_log_prob_list) < beam_size:
                log_probs_topk, word_indices_topk = torch.topk(temp_log_prob_list, len(temp_log_prob_list), dim=-1)
            else:
                log_probs_topk, word_indices_topk = torch.topk(temp_log_prob_list, beam_size, dim=-1)
            
            beam_list = [temp_beam_list[index] for index in word_indices_topk]
            hidden_list = [temp_hidden_list[index] for index in word_indices_topk]
            log_prob_list = log_probs_topk
            output_list = [temp_output_list[index] for index in word_indices_topk]
        
        # replace log prob with mean 
        for idx in range(len(log_prob_list)):
            log_prob_list[idx] /= len(beam_list[idx])
        for idx in range(len(done_log_prob_list)):
            done_log_prob_list[idx] /= len(done_list[idx])
        
        beam_list.extend(done_list)
        log_prob_list.extend(done_log_prob_list)

        best_idx = np.argmax(log_prob_list)
        best_beam = beam_list[best_idx]
        if best_beam[0] == self.dictionary.bos():
            best_beam = best_beam[1:]
        if best_beam[-1] == self.dictionary.eos():
            best_beam = best_beam[:-1]
        best_sentence = [self.dictionary[idx] for idx in best_beam]
        return "".join(best_sentence)
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
        outputs = ""
        return outputs


class Seq2SeqModel(BaseModel):

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        # Hint: Use len(dictionary) in __init__
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        self.hidden_size = hidden_size = args.hidden_size
        self.num_layers = num_layers = args.num_layers
        self.embedding_dim = embedding_dim = args.embedding_dim
        self.vocab_size = len(dictionary)

        self.embedding = nn.Embedding(len(dictionary), embedding_dim)

        self.encoder = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)

        self.attention = getattr(args, "attention", False)

        if self.attention:
            self.decoder = nn.LSTM(embedding_dim + hidden_size, hidden_size, num_layers, batch_first=True)
            self.fc_head = nn.Linear(hidden_size, len(dictionary))
            self.fc_attn_key = nn.Linear(hidden_size, hidden_size)
        else:
            self.decoder = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
            self.fc_head = nn.Linear(hidden_size, len(dictionary))
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################O

    def logits(self, source, prev_outputs, **unused):
        """
        Compute the logits for the given source and previous outputs.

        Args:
            source: The input data.
            prev_outputs: The previous outputs.
            **unused: Additional unused arguments.

        Returns:
            logits: The computed logits.
        """
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        source_embedded = self.embedding(source)
        prev_outputs_embedded = self.embedding(prev_outputs)

        encoder_output, (hidden_state, cell_state) = self.encoder(source_embedded)
        # encoder_output: [batch_size, seq_len, hidden_size]
        # hidden_state, cell_state: [num_layers, batch_size, hidden_size]
        if not self.attention:
            decoder_output, (hidden_state, cell_state) = self.decoder(prev_outputs_embedded, (hidden_state, cell_state))
            # decoder_output: [batch_size, seq_len, hidden_size]
            logits = self.fc_head(decoder_output)
            # logits: [batch_size, seq_len, vocab_size]
        else:
            logits = torch.empty(prev_outputs.shape[0], prev_outputs.shape[1], self.vocab_size, device=source.device)
            for word_idx in range(prev_outputs_embedded.shape[1]):
                prev_word_embedded = prev_outputs_embedded[:, word_idx, :].unsqueeze(1)
                # prev_word_embedded: [batch_size, seq_len=1, embedding_dim]

                query = hidden_state[-1].unsqueeze(1)
                # query: [batch_size, seq_len=1, hidden_size]
                key = self.fc_attn_key(encoder_output)
                # key: [batch_size, seq_len, hidden_size]
                attn_scores = torch.matmul(query, key.transpose(1, 2))
                # attn_scores: [batch_size, seq_len=1, seq_len]
                attn_probs = F.softmax(attn_scores, dim=-1)
                context = torch.matmul(attn_probs, encoder_output)
                # context: [batch_size, seq_len=1, hidden_size]

                inputs = torch.cat([prev_word_embedded, context], dim=-1)
                # inputs: [batch_size, seq_len=1, embedding_dim+hidden_size]
                decoder_output, (hidden_state, cell_state) = self.decoder(inputs, (hidden_state, cell_state))
                # decoder_output: [batch_size, seq_len=1, hidden_size]

                logits[:, word_idx, :] = self.fc_head(decoder_output).squeeze(1)
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
        return logits

    def get_loss(self, source, prev_outputs, target, reduce=True, **unused):
        logits = self.logits(source, prev_outputs)
        lprobs = F.log_softmax(logits, dim=-1).view(-1, logits.size(-1))
        return F.nll_loss(
            lprobs,
            target.view(-1),
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

    @torch.no_grad()
    def generate(self, inputs, max_len=100, beam_size=None):
        """
        Generate text using the trained sequence-to-sequence model with beam search.

        Args:
            inputs (str): The input text, e.g., "改革春风吹满地".
            max_len (int, optional): The maximum length of the generated text.
                                     Defaults to 100.
            beam_size (int, optional): The beam size for beam search. Defaults to None.

        Returns:
            outputs (str): The generated text, e.g., "复兴政策暖万家".
        """
        # Hint: Use dictionary.encode_line and dictionary.bos() or dictionary.eos()
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        self.eval()
        encoded_inputs = self.dictionary.encode_line(inputs, append_eos=False)
        encoded_inputs = torch.cat([torch.tensor([self.dictionary.bos()]), encoded_inputs])
        encoded_inputs = encoded_inputs.to(next(self.parameters()).device).unsqueeze(0)
        # add batch dimension [batch_size=1, seq_len]

        embedded_inputs = self.embedding(encoded_inputs) # [batch_size=1, seq_len, embedding_dim]

        if beam_size is None:
            beam_size = 1

        encoder_output, (hidden_state, cell_state) = self.encoder(embedded_inputs)

        # list of candidate word indices
        beam_list = [[self.dictionary.bos()]] # beam_list: [beam_size, seq_len=1]
        hidden_list = [(hidden_state, cell_state)] # hidden_list: [beam_size, (h, c)]
        log_prob_list = [0] # log_prob_list: [beam_size]

        done_list = []
        done_log_prob_list = []

        for idx in range(max_len):
            temp_beam_list = []
            temp_hidden_list = []
            temp_log_prob_list = []

            if len(beam_list) == 0:
                break

            for beam_idx, beam in enumerate(beam_list):
                current_hidden = hidden_list[beam_idx]
                current_word = torch.tensor([beam[-1]]).unsqueeze(0).to(next(self.parameters()).device)
                # current_word: [1]
                current_word_embedded = self.embedding(current_word)
                # current_word_embedded: [1, embedding_dim]

                if not self.attention:
                    inputs = current_word_embedded.unsqueeze(0)
                    # inputs: [1, 1, embedding_dim]
                else:
                    query = current_hidden[0][-1].unsqueeze(1)
                    # query: [1, 1, hidden_size]
                    key = self.fc_attn_key(encoder_output)
                    # key: [batch_size=1, seq_len, hidden_size]
                    attn_scores = torch.matmul(query, key.transpose(1, 2))
                    # attn_scores: [1, batch_size=1, seq_len]
                    attn_probs = F.softmax(attn_scores, dim=-1)
                    context = torch.matmul(attn_probs, encoder_output)
                    # context: [1, batch_size=1, hidden_size]

                    inputs = torch.cat([current_word_embedded, context], dim=-1)
                    # inputs: [1, 1, embedding_dim+hidden_size]

                decoder_output, (hidden_state, cell_state) = self.decoder(inputs, current_hidden)
                # assert decoder_output.shape == (1, 1, self.hidden_size)
                # decoder_output: [1, 1, hidden_size]

                logits = self.fc_head(decoder_output.squeeze(1)).squeeze(0)

                log_probs = F.log_softmax(logits, dim=-1) # log_probs: [vocab_size]

                log_probs_topk, word_indices_topk = torch.topk(log_probs.squeeze(0), beam_size, dim=-1)

                for word_idx, log_prob in zip(word_indices_topk, log_probs_topk):
                    if word_idx.item() == self.dictionary.eos():
                        done_list.append(beam + [word_idx.item()])
                        done_log_prob_list.append(log_prob.item() + log_prob_list[beam_idx])
                        continue
                    
                    temp_beam_list.append(beam + [word_idx.item()])
                    temp_hidden_list.append((hidden_state, cell_state))
                    temp_log_prob_list.append(log_prob.item() + log_prob_list[beam_idx])

            if len(temp_log_prob_list) < beam_size:
                log_probs_topk, word_indices_topk = torch.topk(torch.tensor(temp_log_prob_list), len(temp_log_prob_list), dim=-1)
            else:
                log_probs_topk, word_indices_topk = torch.topk(torch.tensor(temp_log_prob_list), beam_size, dim=-1)
            
            beam_list = [temp_beam_list[index] for index in word_indices_topk]
            hidden_list = [temp_hidden_list[index] for index in word_indices_topk]
            log_prob_list = [temp_log_prob_list[index] for index in word_indices_topk]

        beam_list.extend(done_list)
        log_prob_list.extend(done_log_prob_list)

        # replace log prob with mean
        for idx in range(len(log_prob_list)):
            log_prob_list[idx] /= len(beam_list[idx])
        
        best_idx = np.argmax(log_prob_list)
        best_beam = beam_list[best_idx]
        if best_beam[0] == self.dictionary.bos():
            best_beam = best_beam[1:]
        if best_beam[-1] == self.dictionary.eos():
            best_beam = best_beam[:-1]
        best_sentence = [self.dictionary[idx] for idx in best_beam]
        return "".join(best_sentence)
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
        outputs = ""
        return outputs
