# Synthesis Experiments

1. Follow [this tutorial](https://pytorch.org/audio/stable/tutorials/tacotron2_pipeline_tutorial.html) to see if tacotron2 can work.
1. Make sure to use a differentiable (neural) vocoder.
1. Create a model based on tacotron2 + vocoder which accepts spectrograms as input and outputs sounds.
1. Train it on pure tone database to see if it can learn to make a few different tones

*Go back and finish clustering experiments*

1. Put the whole thing together: sample the dataset from datapoints in the encoder's embedding space,
   feed them into the synthesizer model (untrained). Backprop using frozen vocoder (assuming vocoder is pretrained - should be).
   Loss function is Euclidean distance between encoder(synth_output) and datapoint from embedding space.