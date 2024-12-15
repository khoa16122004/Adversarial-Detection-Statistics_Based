# Why dont we just using distribution distances with threshold
- When we recive a entire dataset of all adversarial examples, it's of course has the high distributuion distance, but how about there just have a part of dataset is adversarial examples? the distribution distance are low.


# Two samples hypothesis test



| Transform method   | MMD                       | 
|--------------------|---------------------------|
| Flips              |       0.00049             |
| Gaussian Blurring  |       0.00088             |
| Subsampling        |       0.00312             |
| **FGSM**           |       **0.16246**         |
| **PGD**            |       **0.12085**         |


# Experiment
- What is the minimum samples size enough to detect adversarial examples from dataset?

- What is the size of adversarial examples enough to detect adversarial examples from dataset?



# MMD

100%: 0.12646
50%: 0.010474
40%: 0.005703
30%: 0.004279
20%: 0.002019
15%: 0.001445
10$: 0.000666