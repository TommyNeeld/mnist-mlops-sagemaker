The `mnist.json` has already been preprocessed, so it can be used
at inference directly.

Obviously the preprocessing should match how that data was preprocessed
as part of the training.

If you data needs preprocessing, you can also perform the prepreocessing
as part of the Batch Transform Job in `run-batch-transform-job.py`,
you just need to make sure that the data supplied to the Batch
Transform Job has been fully preprocessed as the Transform Job will
only do inference, not preprocessing.
