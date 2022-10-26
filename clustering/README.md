# Clustering Experiments

1. Create a classifier capable of classifying a few pure tone spectrograms
1. Create a classifier capable of classifying a few pure tone spectrograms **+ noise**
1. Create an autoencoder that is capable of reconstructing pure tone spectrograms
1. Create an autoencoder that is capable of reconstructing pure tone spectrograms **+ noise**
1. Create an autoencoder that embeds clusters of similar tones near one another
    -> Create a few classes of tones
    -> Each tone should have small amount of random noise added to it
    -> Create a network that has two loss terms: a reconstruction error and a [clustering error](#clustering-error-term)
    -> Visualize the clustering space

*Then go do the experiments in synthesis*

1. Create a database of vowels: /a/ and /u/
1. See if the autoencoder can cluster it (visualize the cluster space)
1. Create a database of vowels: /a/ and /u/ (male) and /a/ and /u/ (female)
1. See if the autoencoder can cluster it (visualize the cluster space)

## Clustering Error Term

A few loss functions to try:

* [Locality Preserving Loss](https://arxiv.org/pdf/2004.03734.pdf)
* [Group Sparse Regularization](https://arxiv.org/pdf/1607.00485.pdf)
* [Anything from clustering with deep learning](https://www.arxiv-vanity.com/papers/1801.07648/)