# DIMBA: Discretely Masked Black-Box Attack in Single Object Tracking

##
The adversarial attack can force a CNN-based model to produce an incorrect output by craftily manipulating human-imperceptible input. Exploring such perturbations can help us gain a deeper understanding of the vulnerability of neural networks, and provide robustness to deep learning against miscellaneous adversaries. Despite extensive studies focusing on the robustness of image, audio, and NLP, works on adversarial examples of visual object tracking—especially in a black-box manner—are quite lacking. In this paper, we propose a novel adversarial attack method to generate noises for single object tracking under black-box settings, where perturbations are merely added on initialized frames of tracking sequences, which is difficult to be noticed from the perspective of a whole video clip. Specifically, we divide our algorithm into three components and exploit reinforcement learning for localizing important frame patches precisely while reducing unnecessary computational queries overhead. Compared to existing techniques, our method requires less time to perturb videos, but to manipulate competitive or even better adversarial performance. We test our algorithm in both long-term and short-term datasets, including OTB100, VOT2018, UAV123, and LaSOT. Extensive experiments demonstrate the effectiveness of our method on three mainstream types of trackers: discrimination, Siamese-based, and reinforcement learning-based trackers. We release our attack tool, DIMBA, via GitHub https://github.com/TrustAI/DIMBA for use by the community.


##

### Note: This work is accepted by Machine Learning Journal. Pls find the paper here: [DIMBA: Discretely Masked Black-Box Attack in Single Object Tracking](https://link.springer.com/article/10.1007/s10994-022-06252-2)


-- Xiangyu Yin & Wenjie Ruan
