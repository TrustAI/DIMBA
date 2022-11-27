# DIMBA: Discretely Masked Black-Box Attack in Single Object Tracking

##
Deep neural networks (DNNs) are known to be vulnerable to adversarial geometric transformation. This paper aims to verify the robustness of large-scale DNNs against the combination of multiple geometric transformations with a provable guarantee. Given a set of transformations (e.g., rotation, scaling, etc.), we develop GeoRobust, a black-box robustness analyser built upon a novel branch-and-bound search strategy, for determining the worst-case combination of transformations that affect and even alter a network’s output. GeoRobust can provide provable guarantees on finding the worst-case combination based on recent advances in Lipschitzian theory. Due to its black-box nature, GeoRobust can be deployed on large-scale DNNs regardless of their architectures, activation functions, and the number of neurons. Furthermore, with
the proposed parallelisation strategy, on average, GeoRobust takes only 10 seconds to locate the worst-case geometric transformation for ResNet model on ImageNet. We systematically examine 18 ImageNet classifiers, including ResNet family and vision transformers. Our experiments reveal a positive correlation between the geometric robustness of the networks and the parameter numbers. We also observe that increasing the depth of DNN is more beneficial than increasing its width in terms of improving its geometric robustness.

### Note: This work is accepted by Machine Learning Journal. Pls find the paper here: [DIMBA: Discretely Masked Black-Box Attack in Single Object Tracking](https://link.springer.com/article/10.1007/s10994-022-06252-2)


-- Xiangyu Yin & Wenjie Ruan
