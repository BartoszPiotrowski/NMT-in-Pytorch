import random
import time
import sys

from utils import timeSince
from prepare_data import prepareData, tensorsFromPair, tensorFromSentence

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


MAX_LENGTH = 20 # TODO move this


class Encoder(nn.Module):
    def __init__(self, n_words, hidden_dim):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(n_words, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, input_sentence):
        # input is a sentence as a tensor of word indices
        input_sentence_embedded = self.embedding(input_sentence)
        # input for gru is a batch; our batch consists of one example
        input_sentence_embedded.unsqueeze_(0)
        outputs, hidden_state = self.gru(input_sentence_embedded)
        return outputs, hidden_state


class Decoder(nn.Module):
    def __init__(self, hidden_dim, n_words, SOS_token, EOS_token):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(n_words, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, n_words)
        self.softmax = nn.LogSoftmax(dim=2) # TODO why not normal softmax?
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token

    def forward(self, target_sentence_prep, init_hidden_state):
        target_sentence_prep_embedded = self.embedding(target_sentence_prep)
        target_sentence_prep_embedded = F.relu(target_sentence_prep_embedded)
        # input for gru is a batch; our batch consists of one example
        target_sentence_prep_embedded.unsqueeze_(0)
        outputs, hidden_state = self.gru(target_sentence_prep_embedded, init_hidden_state)
        outputs = self.softmax(self.out(outputs))
        return torch.squeeze(outputs, 0), hidden_state

    def evaluate(self, init_hidden_state, max_length):
        outputs = []
        init_input = torch.tensor([[self.SOS_token]])
        init_input_embed = self.embedding(init_input)
        init_input_embed = F.relu(init_input_embed)
        hidden_state = init_hidden_state
        output = init_input_embed
        for _ in range(max_length):
            output, hidden_state = self.gru(output, hidden_state)
            outputSoftmax = self.softmax(self.out(output))
            topv, topi = outputSoftmax.topk(1)
            outputs.append(topi.item())
            if topi.item() == self.EOS_token:
                break
        return outputs


def trainOneSentence(input_sentence, target_sentence,
          encoder, decoder,
          encoder_optimizer, decoder_optimizer,
          criterion, SOS_token, EOS_token,
          max_length=MAX_LENGTH, teacher_forcing=True):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    encoder_outputs, encoder_hidden = encoder(input_sentence)
    # decoder_input is for teacher forcing; SOS token is prepended, last unnecessary
    # element removed
    decoder_input = torch.cat((torch.tensor([SOS_token]), target_sentence), 0)[:-1]
    decoder_outputs, decoder_hidden = decoder(decoder_input, encoder_hidden)
    loss += criterion(decoder_outputs, target_sentence) # TODO probably to change

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / len(target_sentence)


def train(pairs, encoder, decoder, input_lang, output_lang,
          epochs, learning_rate=0.05):

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for i in range(epochs):
        for pair in pairs:
            input_sentence = pair[0]
            target_sentence = pair[1]
            losses = []
            loss = trainOneSentence(input_sentence, target_sentence,
                         encoder, decoder,
                         encoder_optimizer, decoder_optimizer,
                         criterion, input_lang.SOS_token, input_lang.EOS_token)
            losses.append(loss)

        print("Loss: {}\n".format(sum(losses) / len(losses)))
        print("Loss: {}\n".format(sum(losses) / len(losses)))
        evaluateRandomly(pairs, encoder, decoder, input_lang, output_lang)


def evaluate(encoder, decoder, input_sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        _, encoder_hidden = encoder(input_sentence)
        decoder_outputs = decoder.evaluate(encoder_hidden, max_length)
        decoded_words = [output_lang.index2word[o] for o in decoder_outputs]
        return decoded_words


def evaluateRandomly(pairs, encoder, decoder, input_lang, output_lang, n=5):
    for i in range(n):
        pair = random.choice(pairs)
        print("Input: {}".format(
            ' '.join([input_lang.index2word[i.item()] for i in pair[0]])))
        print("Target: {}".format(
            ' '.join([output_lang.index2word[i.item()] for i in pair[1]])))
        output_words = evaluate(encoder, decoder, pair[0])
        print("NMT: {}\n".format(' '.join(output_words)))


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
hidden_dim = 256
encoder1 = Encoder(input_lang.n_words, hidden_dim)
decoder1 = Decoder(hidden_dim, output_lang.n_words,
                   output_lang.SOS_token, output_lang.EOS_token)
train(pairs, encoder1, decoder1, input_lang, output_lang, epochs=1)
evaluateRandomly(pairs, encoder1, decoder1, input_lang, output_lang)
