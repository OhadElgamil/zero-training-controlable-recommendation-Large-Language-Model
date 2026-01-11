Large language model-based recommendation sys-
tems (LLMRec) have demonstrated remarkable performance by
leveraging powerful language representations to predict user
preferences. A growing body of research is now examining the
inner mechanisms of LLMs to identify how specific semantic
or behavioral directions are encoded in their embedding space.
Such directions can be used to manipulate and affect the resulting
recommendations of the model. In this work, we present a novel
method for identifying and applying these vectors to the models,
enabling a controlled way of manipulating recommendation
scores of targeted items without retraining the entire model and
without affecting its overall performance. We use this method
to enhance the recommendation rate of specific items, such
as promoting new products, correcting underexposed content,
or conducting controlled experiments. By leveraging targeted
embedding perturbations, we introduce a method to subtly shift
an item representation in the embedding space to a pre-located
area where items are more likely to be recommended
