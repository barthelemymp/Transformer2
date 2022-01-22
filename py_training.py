# Using Transformer Built-in : https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
import sys

sys.path.append("")
from model_Trans import * 
from utils import *





# Model hyperparameters--> CAN BE CHANGED
batch_size = 10
num_heads = 5
num_encoder_layers = 6
num_decoder_layers = 6
dropout = 0.10
forward_expansion = 2048

#EPOCHS 
num_epochs = 800 

# Tensorboard to get nice loss plot
tf_board_dir = "graph_ddi"
writer = SummaryWriter(tf_board_dir)

#If you want to save the model, u can use this additional parameter 
#to save those better than 
#min_mean_loss = 1.2

#Dataset
train_path = 'train_ddi.csv'
val_path = 'val_ddi.csv'
test_path = 'test_ddi.csv'

#number of amino acids in the input sequence
len_input = 101 
len_output = 118  

#add 2 for start and end token 
len_input +=2
len_output +=2



protein = Field(init_token="<sos>", eos_token="<eos>")
protein_trans = Field(init_token="<sos>", eos_token="<eos>")

data_fields = [('src', protein), ('trg', protein_trans)]
train,val,test = TabularDataset.splits(path='./', train=train_path , validation=val_path , test=test_path ,format='csv', fields=data_fields)
protein.build_vocab(train, min_freq=1)
protein_trans.build_vocab(train,  min_freq=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train, val, test),
    batch_size=batch_size,
    device=device,
    sort = False,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    
)



# Model hyperparameters
src_vocab_size = len(protein.vocab) 
trg_vocab_size = len(protein_trans.vocab) 
assert src_vocab_size==trg_vocab_size, "the input vocabulary differs from output voc"
assert src_vocab_size<=25, "Something wrong with length of vocab in input s."
assert trg_vocab_size<=25, "Something wrong with length of vocab in output s."

embedding_size = len(protein.vocab) #it should be 25. 21 amino, 2 start and end sequence, 1 for pad, and 1 for unknown token
src_pad_idx = protein.vocab.stoi["<pad>"] 


model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    len_input,
    len_output,
    device,
).to(device)

#whyyy 'cpu?'

step = 0
step_ev = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



learning_rate = 3e-4

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_normal_(p)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

# The following line defines the pad token that is not taken into 
# consideration in the computation of the loss - cross entropy
pad_idx = protein.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

load_model= False
save_model= False
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

# In case we would like to see just the Bleu score of the loaded model:
# score = bleu(test_data, model, protein, protein_trans, device)
# print(f"Bleu score {score * 100:.2f}")
# import sys
# sys.exit()
#sentence = "- - - - F V N R W V H Q M K T P L S V I Q L T L D E V E E P A E H I R E E L E R I R K G L E R - - - - - - - - - - - - - - - - -"

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        #step = 0
        #model.eval()
        #translated_sentence = translate_sentence(
             #model, sentence, protein, protein_trans, device, max_length=114
        #)

    #print(f"Translated example sentence: \n {translated_sentence}")
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target[:-1, :])

        output = output.reshape(-1, output.shape[2]) #keep last dimension
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) 

        # Gradient descent step
        optimizer.step()
        

        # plot to tensorboard

    mean_loss = sum(losses) / len(losses)
    writer.add_scalar("Training loss", mean_loss, global_step=step)
    step += 1

    scheduler.step(mean_loss)

    model.eval()
    #translated_sentence = translate_sentence(
         #model, sentence, protein, protein_trans, device, max_length=114
    #)

    #print(f"Translated example sentence: \n {translated_sentence}")
    losses_eval = []

    for batch_idx, batch in enumerate(valid_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target[:-1, :])

        output = output.reshape(-1, output.shape[2]) #keep last dimension
        target = target[1:].reshape(-1)


        loss_eval = criterion(output, target)

        losses_eval.append(loss_eval.item())
        
    mean_loss = sum(losses_eval) / len(losses_eval)
    writer.add_scalar("Validation loss", mean_loss, global_step=step_ev)
    step_ev +=1

    #To save model configuration for which the loss is the lowest on the validation set:
    #if mean_loss < min_mean_loss:
        #min_mean_loss = mean_loss
        
        #checkpoint = {
            #"state_dict": model.state_dict(),
            #"optimizer": optimizer.state_dict(),
        #}
        #save_checkpoint(checkpoint)
        
