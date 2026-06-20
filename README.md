# Mask-guided Semantic Alignment: Robust Learning with Noisy Labels via Temporal Attention Stability
Official PyTorch Implementation of Mask-guided Semantic Alignment: Robust Learning with Noisy Labels via Temporal Attention Stability

## Abstract

![image](.\figs\Overview.png)

Noisy labels pose a significant challenge in training deep neural networks, as corrupted annotations inevitably induce overfitting and significantly degrade generalization. However, existing methods typically rely on static metrics and overlook training dynamics, struggling to effectively separate clean samples from noisy ones under complex and high noise scenarios. In this paper, we propose a temporal attention-based framework for learning with noisy labels. Specifically, we introduce an attention temporal consistency module for reliable sample selection, which separates samples by quantifying the evolutionary stability of visual attention during training. Then, to address the scarcity of clean samples in high-noise scenarios, we design an adaptive attention masks module to synthesize images from clean and noisy samples, which employs differential masking strategies to generate high-quality training samples. Finally, we present a mask-guided feature alignment module, which introduces feature consistency regularization to facilitate the learning of robust representations for noisy data. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches, achieving accuracy gains of 1.6%/2.8% on CIFAR-10/100 under 90% symmetric noise, respectively.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/iCAN-SZU/LOND-DRS/blob/main/LICENSE) file for details.