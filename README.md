# Little is Enough: Boosting Privacy by Sharing Only Hard Labels in Federated Semi-Supervised Learning

## [FedCT](#FedCT)
This repository includes the implementation of [Federated Co-Training Approach FedCT.](https://arxiv.org/pdf/2310.05696.pdf) In this paper, we propose to use federated co-training by having clients share predictions on an unlabeled public dataset iteratively. The server then creates a consensus based on these predictions, which clients utilize as pseudo-labels. In summary, our contributions are:

- (1) a novel federated co-training (FedCT) approach to collaboratively train models from privacy-sensitive distributed data sources via hard label sharing on a public unlabeled dataset that achieves model quality comparable to standard federated learning and distributed distillation and can seamlessly integrate any supervised learning method on clients in the federated system, including interpretable models, such as XGboost, decision trees, Random forest, and rule ensembles.
- (2) Theoretical analysis providing a novel convergence result for hard label sharing and a sensitivity bound for hard label sharing by on-average-leave-one-out-stable machine learning algorithms. This analysis provides Differential Privacy (DP) guarantees for FedCT using the XOR-mechanism.
- (3) Extensive empirical evaluation demonstrating that FedCT achieves a favorable privacy-utility trade-off compared to model parameter and soft label sharing.



## [How to Run an Experiment](#How-to-Run-an-Experiment)
To run an experiment, you have to setup [RunExp.sh](https://github.com/kampmichael/distributedcotraining/blob/main/RunExp.sh) file with your desired parameters and then use `bash RunExp.sh` to start the experiment.
<!---
## [Citation](#citation)
If you use our work, please cite the following paper:

```bibtex
@article{abourayya2023protecting,
  title={Protecting Sensitive Data through Federated Co-Training},
  author={Abourayya, Amr and Kleesiek, Jens and Rao, Kanishka and Ayday, Erman and Rao, Bharat and Webb, Geoff and Kamp, Michael},
  journal={arXiv preprint arXiv:2310.05696},
  year={2023}



