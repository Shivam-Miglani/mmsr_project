# Tag 8 million Youtube video

The YouTube8M challenge is a multi-class classification problem, where we are asked to predict for each video, given video & frame level audio and frame RGB features, to which group of categories it belongs to.
The number of classes is 3807 for our subset of data.


## Approach

`classifier.ipynb` shows the approach to solve the problem. 
The main idea is to separate `video level features` with `frame level features`, and apply context gating (non linear learnable unit to model interdependencies between activations) [1] for video classification

### Video level features
`mean rgb` and `mean audio` are the video level features. We pass them through `Dense` layers.

### Frame level features
`mean frame rgb` and `mean frame audio` are the frame the frame level feautures. We pass them through `Bi-LSTM` layers.

### Merge 
In the end we merge the outputs of video and frame level features into a dense layer and a sigmoid layer is used to predict the `tag` for the video.


[1] Miech, Antoine, Ivan Laptev, and Josef Sivic. "Learnable pooling with Context Gating for video classification." arXiv preprint arXiv:1706.06905 (2017).

