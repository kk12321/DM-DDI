# DM-DDI
worthy of note：

1、Before you start running the code, you need to generate the pre-trained embedding files  by the pretrain.py file  in /data, and select the corresponding relationship files in the /graph  directory.

2、After experimental investigation, it is found that the experimental effect is best when three fusions are carried out, so this paper sets up 4 layers with nodes (the number of nodes in each layer is 1716, 2000, 256, 65, respectively)

3、The default is to test for 65 categories of DDI data, when you need to test different scale of ddi class of data, you need to modify the corresponding parameters.

4、after obtained the DM-DDI embeddings,fed it to DNN classfier,just as described in the article.

5、our experiment run on Server ubantu 16,32 cpu,256G.