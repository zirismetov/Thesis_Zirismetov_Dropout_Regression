# Thesis_Zirismetov_Dropout_Regression
python taskgen.py -dataset weatherHistory CalCOFI -dropoutModule advancedDropout gaussianDropout simpleDropout dropConnect noDrop -lr 0.1 0.01 0.001 -test_size 0.20 0.33 0.50 -batch_size 128 256 -layers_size 1,32,32,32,1 1,32,64,32,1 1,64,64,64,1 1,64,128,64,1 1,128,128,128,1 -drop_p 0,0.5,0.5 0.2,0.5,0.5 0.2,0.2,0.2 -epoch 5 10 15
