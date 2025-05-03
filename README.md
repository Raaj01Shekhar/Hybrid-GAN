# Hybrid-GAN
Generative Adversarial Networks (GANs), introduced by Goodfellow et al. (2014), have revolutionized the domain of synthetic image generation by framing the problem as a minimax game between a generator and a discriminator. In the medical imaging domain, GANs have been widely adopted for data augmentation, anomaly detection, and image-to-image translation due to their ability to learn complex data distributions without explicit supervision (Yi et al., 2019). However, training stability and mode collapse remain significant challenges in vanilla GANs, especially when applied to high-resolution and domain-specific datasets such as brain MRI scans.

To address these limitations, our approach leverages a hybrid GAN architecture that combines the structural design of a Deep Convolutional GAN (DCGAN) with the robustness of a Conditional Wasserstein GAN with Gradient Penalty (cWGAN-GP). DCGANs are known for their effective use of transposed convolutions and batch normalization, which stabilize training and enable the generation of visually coherent images (Radford et al., 2015). On the other hand, WGAN-GP replaces the original binary cross-entropy loss with a Wasserstein loss, which provides a smoother gradient landscape and improves convergence, especially in scenarios with limited data (Gulrajani et al., 2017). By incorporating class conditioning, the generator and discriminator are both guided by categorical tumor labels, enabling precise control over the type of image being synthesized.

Medical image datasets, such as those involving brain tumors, often suffer from class imbalance and limited annotated data, which can degrade the performance of supervised learning models. Hybrid GANs offer a viable solution by generating class-specific synthetic images that augment the training set, thereby enhancing model generalization and diagnostic accuracy. The class-conditioning mechanism ensures that the synthesized images not only retain the anatomical realism but also reflect the pathological characteristics unique to each tumor type (Frid-Adar et al., 2018). Furthermore, incorporating metrics such as SSIM, PSNR, entropy, histogram similarity, and Wasserstein distance allows for quantitative validation of image fidelity and diversity, ensuring that generated samples are both realistic and useful for downstream tasks.

By integrating these methodologies, our hybrid GAN aims to address the dual challenge of data scarcity and class imbalance in medical imaging, with a particular focus on high-resolution grayscale brain MRI synthesis. This framework can be extended to other imaging modalities and conditions, laying the foundation for more robust and explainable AI-driven diagnostics in healthcare.

@misc{hybridgan2025,
  title={Hybrid GAN for Brain Tumor MRI Augmentation},
  author={Raaj Shekhar},
  year={2025},
  note={https://github.com/Raaj01Shekhar/hybrid-gan}
}
