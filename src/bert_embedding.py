from transformers import BertTokenizer, BertModel
import torch
import numpy as np


def encode_sentences(sentence, tokenizer, max_length=64):
    # Encode the sentence
    encoded = tokenizer.encode_plus(
        text=sentence,  # the sentence to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length=1024,  # maximum length of a sentence
        pad_to_max_length=True,  # Add [PAD]s
        return_attention_mask=True,  # Generate the attention mask
        return_tensors='pt',  # ask the function to return PyTorch tensors
        return_token_type_ids=False
        # For having embeddings of a sentence we don't need it. However if we want to go with QA we need this.
    )

    # Get the input IDs and attention mask in tensor format
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    return input_ids, attention_mask


def get_bert_features(text=""):
    text = "Here is a sample sentence, which I want its embeddings."
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Initializing the Tokenizer
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True,
                                      # Whether the model returns all hidden-states
                                      )
    input_ids, attention_mask = encode_sentences(text, tokenizer=tokenizer, max_length=64)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # last_hidden_state = outputs[0]
    # pooler_output = outputs[1]
    hidden_states = outputs[2]

    ### for finding subtokens and make it compatible with EEG ###
    textt = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(textt)
    subtoken_indices = [i for i in range(len(tokenized_text)) if "#" in tokenized_text[i]]
    ##################################################################

    compatible_hidden_states = BERT_EEG_compatibility(hidden_states=hidden_states, subtoken_indices=subtoken_indices)
    return compatible_hidden_states
    # hidden-states of the model at the output of each layer + the initial embedding outputs (so, it includes 12 + 1 = 13 tensors)
    # It is in format of Numpy Arrays for ease of use with Ridge Regression
    # It is not in format of Pytorch Tensors


def get_bert_features_for_layer(text="", layer=12):
    bert_features_for_all_layers = get_bert_features(text)
    return bert_features_for_all_layers[layer]


def BERT_EEG_compatibility(hidden_states, subtoken_indices):
    hidden_states_numpy = []
    for hidden_state in hidden_states:
        hidden_state_numpy = hidden_state[0].cpu().detach().numpy()
        root_index = 0
        previous_subtoken = 0
        new_token = False
        total_subtokens = 1
        for subtoken_index in subtoken_indices:
            # ----- Just an initialization for the first subtoken -----
            if previous_subtoken == 0:
                previous_subtoken = subtoken_index
            if root_index == 0:
                root_index = subtoken_index - 1
            # --------------------------------

            if subtoken_index - previous_subtoken > 1:
                ### We are moving to a new token.
                ### First, we Average the previous token and it's subs embedding
                ### Then, we delete those subtokens from output (To use EEG Representation)
                hidden_state_numpy[root_index] = hidden_state_numpy[root_index] / total_subtokens
                for i in range(1, total_subtokens):
                    # Removing Subtokens embedding after averaging the root node from output and insert a list of zeros at the end to mainain the size
                    np.delete(hidden_state_numpy, root_index + 1, 0)
                    np.vstack((hidden_state_numpy, np.random.uniform(-0.5, 0.5, 768)))

                total_subtokens = 2
                root_index = subtoken_index - 1
            else:
                new_token = False
                total_subtokens += 1
                previous_subtoken = subtoken_index
                hidden_state_numpy[root_index] += hidden_state_numpy[subtoken_index]

        for i in range(1, total_subtokens):
            # Removing Subtokens embedding after averaging the root node from output and insert a list of zeros at the end to mainain the size
            np.delete(hidden_state_numpy, root_index + 1, 0)
            np.vstack((hidden_state_numpy, np.random.uniform(-0.5, 0.5, 768)))

        hidden_states_numpy.append(hidden_state_numpy)

    hidden_states_numpy = np.asarray(hidden_states_numpy)
    return hidden_states_numpy

##################### Some Information and Exmaples of how we can use this by Arad  🤗 ############################
# hidden_states[12] == last_hidden_state  # just to verify
# print(len(hidden_states[12]))  # It is the batch size, which is 1 here
# print(len(hidden_states[12][0]))  # It is the size of our padding which is 64
# print(len(hidden_states[12][0][0]))  # It is the number of features BERT dedicate for each word, which is 768
#
# # So in order to have a words embedding, All we need to do is : hidden_states[numebr_of_layer][number_of_batch(0 in our case)][number_of_token]
# # Like this, it is the embedding for the 10th token, from the last layer (12th Layer):
# # -------------------------------------------------------------------------------------
# # Important point to remember Hidden_states[0] iis just the initial embedding outputs.
# # So, Keep in mind that if we want the first layer's output, we should consider this: hidden_states[1] not hidden_states[0]
# # and similarly, when we want the 12th layer's output, we should run : hidden_states[12] not hidden_states[11]
# hidden_states[12][0][9]
#
# # it is the embedding for the 20th token, from the 5th layer:
# hidden_states[5][0][19]
