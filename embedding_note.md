Embedding
=========

Definition: An embedding is a mapping from discrete objects,
such as words, to vectors of real numbers. The individual
dimensions in these vectors typically have no inherent meaning.
Instead, it's the overall patterns of localtion and distance
between vectors that machine learning takes advantage of.


Mini-FAQ
========

**Is "embedding" an action or a thing?** Both. People talk about embedding words in a vector space (action)
and about producing word embeddings (things). Common to both is the notion of embedding as a mapping from
discrete objects to vectors. Creating or applying that mapping is an action, but the mapping itself is a thing.

**Are embeddings high-dimensional or low-dimensional?** It depends. A 300-dimensional vector space of words and phrases,
for instance, is often called low-dimensional (and dense) when compared to the millions of words and phrases it can contain.
But mathematically it is high-dimensional, displaying many properties that are dramatically different from what our human
intuition has learned about 2- and 3-dimensional spaces.

**Is an embedding the same as an embedding layer?** No. An embedding layer is a part of neural network, but an embedding is a more general concept.


.. rubric:: Footnotes

.. [#] `ML Concepts - Embeddings <https://tensorflow.google.cn/guide/embedding>`_
.. [#] `A cool display <http://projector.tensorflow.org>`_
