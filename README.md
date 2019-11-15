# Transformer - Attention Is ALL You Need  
A Tensorflow implementation of Transformer in version 1.12. The core fucntions in transformer such as __scaled dot prodction attention__, __multi-head attention__ and __feedforward network__, were implemented in `nn.py`  

 > For more details, read the paper: Ashish Vaswani, et al. ["Attention is all you need."](http://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) 

Noticed: TF 1.x scripts will not continue to work with TF 2.0. Therefore, the new variants of transformer will move to Tensorflow 2.0.
  
  
[Example](https://towardsdatascience.com/transformer-xl-explained-combining-transformers-and-rnns-into-a-state-of-the-art-language-model-c0cfe9e5a924) for Multi-Head Attention  
![](https://github.com/pjlintw/tf-transformer/blob/master/img/multi-head%20attention%20viz.png)


# Prerequisites
### Dependencies
- TensorFlow >= 1.12
- Python >= 3.6


# Dataset  
    
Noticed: 3000 exampels were used for my experiement. Dataset is not provided.  

For training the model, source and target examples should be provided in `data/src.train.example.txt` and `data/tgt.train.example.txt`. Each source example is corresponding to same index in the target file.

In the reconstruction task, the encoder produces a low-dimension representation by taking source example. The decoder try to reconstruct original sentence by recieved the low-dimension code and target example.

You can replace dataset with parallel corpus for machine translation task. Concretely, the file of sources `src.txt` contains sentences of langauge A. Sentences of language B is in `tgt.txt`. Thing to be noticed: you should provide tow vocaburary files and modify codes for vocaburary creating. 



# Train

### Parameters  

To make sure this code is well implemented and trainable, I trained sentence reconstruction over a tiny Classical Chinese dataset with this repository. Therefore, the parameters were set to overfit on dataset.   
  
  
<!-- mdformat off(no table) -->  
  
| Parameters               | number   | 
| ------------------------ | -------- |
| EPOCH                    | 2000     |
| BACTH SIZE               | 32       |
| DROPOUT                  | 0.1      | 
| NUM LAYERS               | 1        |
| D MODEL                  | 128      |
| NUM HEADS                | 8        |
| ENCODER SEQUENCE LENGTH  | 100      |
| DECODER SEQUENCE LENGTH  | 100      |
  

<!-- mdformat on -->

  

### Training
```
python train_transformer.py
```

# Results
The vallina transformer consist of two attention-based netwoks: encdoer and decoder. That is similar architecture to autoencoder (Hinton & Salakhutdinov, 2006.). The experiement sugguests that transformer can be train on reconstruction task, both for short and long sequence.

short sentence (17 tokens)
```
source         > 秋天，吳國攻伐楚國，楚軍擊敗吳軍。

reconstruction > 秋天，吳國攻伐楚國，楚軍擊敗吳軍。<eos>
```


long sentece (62 tokens)
```
source         > 樊穆仲說：魯懿公之弟稱，莊重恭敬地事奉神靈，敬重老人；處事執法，必定諮詢先王遺訓及前 朝故事；不牴觸先王遺訓，不違背前朝故事。

reconstruction > 樊穆仲說：魯懿公之弟稱，莊重恭敬地事奉神靈，敬重老人；處事執法，必定諮詢先王遺訓及前朝故事；不牴觸先王遺訓，不違背前朝故事。<eos>
```
  
###  
* construct the mask correctly.
* schedule learning rate is must.



### Implementation Reference  
* https://github.com/tensorflow/models/tree/master/official/transformer
* https://github.com/Kyubyong/transformer/
* https://github.com/lilianweng/transformer-tensorflow
* [Transformer model for language understanding](https://www.tensorflow.org/tutorials/text/transformer)  
* [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

