#Use this file if you want to translate the sequence aminoacid by aminoacid (there is error propagation).

n_trans = 200 #number of sequences we want to translate
file_name = "test_real.csv" 

load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
df = pd.read_csv(file_name)
prot_list= df.iloc[:n_trans,0].to_list()
prot_trans_list =  df.iloc[:n_trans,1].to_list()
translation = []
for i in range(n_trans):
	trans_sequence = translate_sentence(model,prot_list[i],protein, protein_trans,device)
	translation.append(" ".join(trans_sequence[:-1]))

frame = pd.DataFrame([prot_trans_list, translation]).T
frame.columns = ['target','output']
frame.to_csv("frame_trans.csv",index=False)
