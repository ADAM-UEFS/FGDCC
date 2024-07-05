# FGDCC: Fine-Grained Deep Cluster Categorization - An Architecture for Intra-Class Variability in Problems in FGVC tasks

## Method

![FGDCC](https://github.com/FalsoMoralista/FGDCC/blob/main/util/images/FGDCC.png)

FGDCC is an architecture developed to tackle intra-class variability problems in FGVC tasks. It operates by performing hierarchical classification of class-wise cluster-assignments conditioned on parent labels (original dataset targets). 
---

### Requirements
* Python 3.8 (or newer)
* PyTorch 2.0
* torchvision
* [Faiss](https://github.com/facebookresearch/faiss)
* Other dependencies: pyyaml, numpy, opencv, submitit, timm

## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

## Other
If you find this repository useful in your research, please consider giving a star :star: and a citation

Based on [I-JEPA](https://github.com/facebookresearch/ijepa)
