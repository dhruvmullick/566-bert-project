from transformers import BertTokenizer, BertModel
import torch


def encode_sentences(sentence, tokenizer, max_length=64):
    # Encode the sentence
    encoded = tokenizer.encode_plus(
        text=sentence,  # the sentence to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length=64,  # maximum length of a sentence
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


def get_BERT_hidden_state(text=""):
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
    return hidden_states
    # hidden-states of the model at the output of each layer + the initial embedding outputs (so, it includes 12 + 1 = 13 tensors)

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
# # 🤗
# hidden_states[12][0][9]
#
# # it is the embedding for the 20th token, from the 5th layer:
# hidden_states[5][0][19]
