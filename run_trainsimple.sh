#/bin/bash

python run_train.py --fam 17 \
		    --nlayer 2 \
		    --embedding_size 55 \
		    --nhead 5 \
		    --batch_size 32 \
		    --forward_expansion 2048 \
		    --num_epochs 5000
		    --dir_save "simpletrans_ddi_17.pth.tar"\ 
