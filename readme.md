This repository hosts the source code for my COMP 4570SEF Capstone Computing Project, completed as the final-year project for the course.

### Abstract
Generative Artificial Intelligence (genAI) has emerged as a revolutionary technology across numerous industries due to its capability to synthesize novel data. At the core of modern genAI are two distinct probabilistic frameworks: **Variational Autoencoders (VAEs)** and **Probabilistic Diffusion Models**. This project explores two fundamental research questions:
1. What are the key mathematical, structural, and performance differences between VAEs and Diffusion Models?
2. How can these generative paradigms be effectively deployed in practical, real-world engineering applications?

The project explores advanced generative modeling techniques in deep learning, with a focus on two state-of-the-art approaches: Variational Autoencoders (VAEs) and Diffusion Models. The implementations provided here correspond directly to the models designed, trained, and evaluated in the accompanying project report.

**Key Contents**
- Complete PyTorch implementations of a Variational Autoencoder (VAE) for latent space learning and generative tasks.
- A Denoising Diffusion Probabilistic Model (DDPM) implementation, including sampling and training pipelines.
- Training and evaluation scripts, configuration files, and Jupyter notebooks for experiments and visualization.
- Pre-trained model checkpoints (where applicable) and sample generated outputs.
This codebase serves as a practical demonstration of the theoretical concepts, architectural design choices, and empirical results discussed in the full project report. It is intended both as a record of the capstone work and as a reusable reference for anyone interested in VAEs, diffusion models, or generative AI techniques.

Using the standard **MNIST benchmark dataset**, this project provides an in-depth investigation into the theoretical and mathematical foundations of both frameworks. It moves beyond academic benchmarks to explore an industrial case study: reconstruction-based anomaly detection for defect inspection using the **MVTec Anomaly Detection (MVTec AD)** dataset.

---

## Project Aims and Objectives
The project is structured around four primary research aims:
1. **Theoretical and Mathematical Investigation:** Elucidate the foundational mechanics of both paradigms, including probabilistic modeling, latent variable inference, the mathematical derivation of the Evidence Lower Bound (ELBO), the reparameterization trick in VAEs, and the forward/reverse stochastic processes in Diffusion Models. It traces the conceptual evolution from seminal frameworks (Kingma & Welling, 2013; Ho et al., 2020; Kingma et al., 2021) and examines inherent limitations, such as the information bottleneck in VAEs and sequential sampling constraints in Diffusion Models.
2. **From-Scratch Implementation:** Develop fully functional, reproducible implementations of VAE and Denoising Diffusion Probabilistic Model (DDPM) variants from scratch using **PyTorch**, validated against the MNIST handwritten digit dataset.
3. **Multi-Dimensional Comparative Analysis:** Systematically evaluate both implemented models across four critical axes under identical hardware conditions:
   * **Latent Space Characteristics:** Structure, continuity, interpretability, and disentanglement.
   * **Generation Quality:** Metric-driven evaluation via Fréchet Inception Distance (FID).
   * **Synthetic Data Utility:** Downstream performance evaluation by training a classifier exclusively on synthetic data and testing it on real-world data.
   * **Sampling Efficiency:** Quantification of generation speed (seconds per image).
4. **Real-World Application Deployment:** Bridge the academic-industrial gap by designing, implementing, and validating a reconstruction-based Industrial Anomaly Detection (IAD) system using the Diffusion Model to inspect defects (e.g., in glass bottles) without requiring anomalous training data.

---

## Methodology & System Architecture

The system architecture spans four main components: a core model implementation engine, localized diagnostic scripting pipelines, a multi-faceted quantitative and qualitative evaluation suite, and a specialized reconstruction-based industrial anomaly detection framework.

---

### 1 VAE Architecture & Pipeline Implementation

<p align="center">
  <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/abfeb2c8-c196-43df-8ff0-376f51570f12" />
  <br>
  <em>The architecture of VAE</em>
</p>

The Variational Autoencoder implementation replicates the foundational probabilistic framework introduced by Kingma and Welling (2014). It intentionally replaces the deterministic bottleneck of classical autoencoders with a probabilistic latent space parameterized by continuous distribution metrics.

```
[Input Data x] ---> [Encoder (φ)] ---> [Mean μ & Log-Variance log(σ²)]
                                                    |
                                           (Reparameterization via ε)
                                                    v
[Output x̂] <--- [Decoder (θ)] <---------------- [Latent Vector z]

```

#### Mathematical Formulation & Optimization

The encoder networks learn to parameterize a variational distribution $Q_\phi(z|x)$ that approximates the intractable true posterior distribution $P(z|x)$ using a diagonal multivariate Gaussian distribution. Conversely, the decoder network parameters ($\theta$) learn to define the conditional likelihood distribution $P_\theta(x|z)$ mapping latent points back into data coordinates.

To enable end-to-end backpropagation through the stochastic bottleneck, the pipeline employs the **reparameterization trick**. Instead of drawing samples directly from the non-differentiable distribution $z \sim Q_\phi(z|x)$, the stochasticity is isolated by introducing an independent noise variable:

$$\epsilon \sim \mathcal{N}(0, I)$$

The latent variable $z$ is then deterministically computed as:

$$z = \mu + \sigma \odot \epsilon$$

The entire architecture optimizes the total Evidence Lower Bound (ELBO) loss function, which splits into two discrete functional operations:

$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{Q_\phi(z|x)}[\log P_\theta(x|z)] - D_{\text{KL}}(Q_\phi(z|x) \parallel P(z))$$

* **Reconstruction Loss:** Evaluated via binary cross-entropy to measure how faithfully the decoded tensor $\hat{x}$ approximates the input tensor $x$.

* **Kullback–Leibler (KL) Divergence:** Functions as an algebraic regularizer to penalize deviations between the learned approximate posterior $Q_\phi(z|x)$ and the standard Gaussian prior $P(z) \sim \mathcal{N}(0, I)$.



#### Network Architecture Specifications

* **Structural Topology:** Both the encoder and decoder are constructed as symmetric 2-layer Deep Neural Networks (DNNs).
  
* **Activation Functions:** Rectified Linear Units (ReLU) handle non-linear activation states across hidden layers.
  
* **Training Schedule:** The network executes over 10 training epochs using the MNIST handwritten digit benchmark dataset.
  
* **Latent Space Dimensionality:** Set explicitly to 2 dimensions for topological visualization tasks, and dynamically configured across 5, 10, 20, 50, and 100 dimensions during latent smoothing evaluations.



---

### 2 Diffusion Model (DDPM) Architecture & Pipeline Implementation

<p align="center">
   <img width="1490" height="598" alt="image" src="https://github.com/user-attachments/assets/35d3f1f9-91d7-47ae-baa1-755cd7a74664" />
  <br>
  <em>The architecture of Diffusion Model</em>
</p>

The Diffusion Model is engineered around the Denoising Diffusion Probabilistic Model (DDPM) framework detailed by Ho et al. (2020), focusing on generating high-fidelity outputs through iterative multi-step synthesis.

#### Forward Process & Noise Scheduling

The model establishes a forward process that incrementally destroys input data structure over a sequence of $T = 1000$ discrete timesteps by injecting Gaussian noise. Drawing from the insights of Kingma et al. (2021) and Chen (2023), the framework adopts an **exponential Signal-to-Noise Ratio (SNR) increase pattern**. This exponential schedule prevents the abrupt structural destruction typical of standard linear schedules during early timesteps, ensuring a more gradual degradation that preserves high-frequency edge details while providing enough noise variance at later stages.

#### Core Denoising Backbone (U-Net)

<p align="center">
  <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/8757d051-95d0-4786-bce0-55b5a999ec20" />
  <br>
  <em>The U-Net Architecture.</em>
</p>

The neural network backbone is structured as a class-conditioned, symmetric U-Net with skip connections to accurately predict the added noise at each timestep.

| Architectural Module | Component Specifications & Structural Layers |
| --- | --- |
| **Encoder Path** | Consists of 3 downsampling blocks (`DownC`), each integrating residual convolutions, sinusoidal time-embedding modulations, and self-attention layers.|
| **Bottleneck** | Configured with 2 mid-processing blocks (`MidC`) that embed deep spatial attention layers.|
| **Decoder Path** | Features upsampling blocks (`UpC`) that concatenate incoming feature maps directly with matching encoder outputs via skip connections.|
| **Conditioning Engine** | Class conditioning is injected via a trainable embedding layer projected jointly with the sinusoidal time-embedding.|
| **Guidance System** | Integrates Classifier-Free Guidance (CFG) by incorporating a null class token to balance sample diversity and fidelity.|


---

### 3 Diagnostic Scripting Components (Visualization & Transitions)

To inspect the structural properties of the VAE's latent space, two specialized diagnostic subroutines were created to run side-by-side with the main models.

#### Latent Manifold Mapping (`Latent_Space_Visualization.py`)
This script checks for a pre-trained model checkpoint or triggers an active training routine via `VAE.py`. It feeds the MNIST test set through the trained encoder in a gradient-free environment to isolate the 2-dimensional latent mean vectors ($\mu$). These coordinates are mapped onto a 2D scatter plot using Matplotlib, color-coded by digit class. The visualization is also used to evaluate the structural effects of modifying individual loss weights (such as running reconstruction-only or KL-only objectives).

#### Topological Manifold Interpolation (`smooth_transition.py`)
This routine validates the geometric continuity of the latent manifold by interpolating between distinct digit endpoints. The script extracts the latent coordinates of two different digit representations ($Z_{\text{start}}$ and $Z_{\text{end}}$) and runs a linear interpolation across a sequence of $N$ frames:

$$Z_{\text{interp}} = (1 - \alpha) \cdot Z_{\text{start}} + \alpha \cdot Z_{\text{end}}$$

The variable $\alpha$ steps continuously from 0 to 1. Each interpolated vector is processed by the decoder network to generate a pixel-space image. These frames are compiled into a horizontal preview plot to observe morphological transitions across the manifold. To track how latent space dimensionality affects structural morphing, the routine tests dimensions across 2, 5, 10, 20, 50, and 100 states for both standard and KL-stripped VAE configurations.

---

### 4 Multi-Faceted Model Analysis Strategy
The performance comparison between the VAE and Diffusion Model relies on a structured analysis strategy across three core testing dimensions.

#### 4.1. Latent Space Structural Characterization
* **Dimensionality & Structure:** The VAE's explicit 2D coordinate clusters are mapped directly via scatter plots to evaluate class separation. Because the Diffusion Model lacks a static low-dimensional bottleneck, its implicit latent structure is monitored by sampling intermediate noisy tensors at fixed intervals during the reverse denoising path.
* **Interpretability:** Handled for the VAE by checking cluster clustering and tracking the semantic clarity of decoded latent trajectories. The Diffusion Model's interpretability is evaluated by compiling its reverse path into sequential time grids to evaluate how structure emerges from noise.
* **Disentanglement:** Evaluated qualitatively in the VAE by checking the boundary overlap between different classes. For the Diffusion Model, it is assessed by tracking whether class-specific features emerge independently during the final stages of the reverse process.



#### 4.2. Quantitative Evaluation Suite

* **Fréchet Inception Distance (FID):** Measures distributional alignment by generating 10,000 synthetic images from both the VAE and Diffusion Model pipelines. These outputs are evaluated against the original MNIST test set to assess generation quality.

<p align="center">
  <img width="600" height="150" alt="image" src="https://github.com/user-attachments/assets/f407f820-4ab1-4ce2-a0d7-8a9a99414a46" />
  <br>
  <em>Synthetic data training process diagram.</em>
</p>


* **Accuracy via Synthetic Data Training:** Evaluates data utility by generating 50,000 synthetic images from each model to build isolated training sets. Labels for VAE data are inferred from latent cluster origins, while Diffusion Model labels match the conditioned class tokens. Separate classifiers with identical architectures are trained on these datasets and tested against the real MNIST test set. A baseline classifier trained on real MNIST data provides a performance benchmark.

#### 4.3. Computational Efficiency Metrics

* **Sampling Speed Analysis:** Measures runtime performance under identical hardware configurations. VAE evaluation tracks the execution time of a single parallelized batch generation of 1,000 images through the decoder. Diffusion Model evaluation tracks the time required to complete the full $T = 1000$ sequential reverse denoising steps to generate 1,000 images.

---

### 5 Industrial Anomaly Detection (IAD) Pipeline Application

To evaluate practical utility beyond standard benchmarks, the project implements a reconstruction-based anomaly detection pipeline based on the methodology of Karsten et al. (2022).

<p align="center">
  <img width="600" height="150" alt="image" src="https://github.com/user-attachments/assets/d44e239c-292a-4b01-808a-c382f4056114" />
  <br>
  <em>Training and detection process diagram.</em>
</p>

* **Target Domain:** Deployed on the industrial MVTec Anomaly Detection dataset, specifically focusing on the glass bottle category.

* **Training & Adaptation:** The underlying DDPM U-Net architecture is adjusted to match the spatial resolution of the MVTec dataset. The model trains exclusively on defect-free normal glass bottle instances, ensuring it learns only the nominal data distribution. Anomalous images are held back entirely for the testing phase.

* **Detection Mechanics:** During inference, a test image is partially corrupted with noise and processed through a reduced reverse execution path of 500 to 600 steps (optimized from the full 1,000 steps to balance real-time throughput and synthesis quality).

* **Localization & Scoring:** The pipeline computes a pixel-wise absolute or squared difference map between the original input frame and the model's clean reconstruction. Because the model has only learned normal patterns, it cannot reconstruct anomalous regions, causing a localized spike in reconstruction error. This difference map is translated into a spatial anomaly heatmap and aggregated into a scalar score to classify the image as normal or anomalous based on a set threshold.

---

## Experimental Results and Analysis
This part provides a detailed analysis of the empirical findings obtained from the implementation, testing, and comparative evaluation of the Variational Autoencoder (VAE) and Denoising Diffusion Probabilistic Model (DDPM) frameworks. The results are structured to evaluate latent space topology, manifold continuity, image generation fidelity, sampling efficiency, and practical utility in an industrial inspection scenario.

---

### 1 Latent Space Visualization & Loss Formulation Mechanics
<p align="center">
  <img width="1015" height="310" alt="image" src="https://github.com/user-attachments/assets/4e883f1b-f4db-403b-8105-776aa131dad7" />
  <br>
  <em>Comparison of 2D Latent Distributions in VAEs Trained with Different Loss Formulations.</em>
</p>

The geometric structure of the learned 2-dimensional latent space in the VAE is heavily dictated by the mathematical trade-off inherent within the Evidence Lower Bound (ELBO) objective function. By utilizing the diagnostic tool `Latent_Space_Visualization.py`, the spatial distribution of the latent mean vectors ($\mu$) across the MNIST dataset was mapped under three distinct loss configurations:

* **Balanced ELBO (Combination Loss):** When the model optimizes both the reconstruction loss (binary cross-entropy) and the Kullback-Leibler (KL) divergence regularizer simultaneously, the latent space achieves an optimal equilibrium. The reconstruction term forces the encoder to generate distinct, information-dense coordinates to permit accurate pixel-level recovery, while the KL term actively regularizes the approximate posterior distribution $q(z|x)$ toward a standard multivariate Gaussian prior $\mathcal{N}(0, I)$. This joint optimization yields well-defined, class-separable clusters with visible, navigable gaps between different digit classes, forming a continuous and semantically organized latent manifold.

* **Reconstruction Loss Only (No-KL Framework):** When the KL divergence regularizer is entirely omitted, the encoder is free to minimize reconstruction error without any structural constraints. It exploits arbitrarily large means and variances, causing class clusters to expand drastically across the latent plane. The data points spread into elongated, comet-like tails and diffuse arms extending out to $\pm 10$ or further. Without the compactness enforced by the KL prior, the latent space suffers from severe overlap in central regions and forms a highly irregular, non-continuous manifold that cannot be reliably sampled.

* **KL Divergence Only (Posterior Collapse):** When the reconstruction loss is completely removed, the objective function forces every input $x$ to map uniformly to the prior distribution, such that $q(z|x) \approx \mathcal{N}(0, I)$. The encoder quickly stops extracting input-specific features and outputs near-zero means and near-unit variances regardless of the input image. This triggers total posterior collapse, compressing all digit classes into a single, dense ball tightly clustered around the origin ($0,0$). All input-specific information and class structures are completely erased, rendering the latent codes uninformative.

---

### 2 Latent Space Smooth Transition & Topological Continuity

To evaluate the mathematical continuity and structural stability of the learned latent manifolds, linear interpolations were executed using `smooth_transition.py`. The script computes intermediate latent vectors ($Z_{\text{interp}}$) between two distinct digit endpoints ($Z_{\text{start}}$ and $Z_{\text{end}}$) across a sequence of $N$ steps using the following formulation:

$$Z_{\text{interp}} = (1 - \alpha) \cdot Z_{\text{start}} + \alpha \cdot Z_{\text{end}}$$

The interpolation variable $\alpha$ steps continuously from 0 to 1. The resulting vectors are passed through the decoder to evaluate structural morphing across multiple latent dimensionalities (2, 5, 10, 20, 50, and 100 dimensions).

#### Morphological Behavior Across Loss Settings

<p align="center">
  <img width="600" height="200" alt="image" src="https://github.com/user-attachments/assets/10eb2a06-cd5c-494a-bd6e-11edc72e078a" />
  <br>
  <em>Comparison of 2D Latent Transition in VAEs Trained with Different Loss Formulations. The latent space of VAE</em>
  <br><br>
  <img width="600" height="200" alt="image" src="https://github.com/user-attachments/assets/b7b94395-e723-4079-9ba9-adc4b7ebd617" />
  <br>
  <em>Comparison of different dimensions of Latent Transition (reconstruction + KL)</em>
  <br><br>
  <img width="600" height="200" alt="image" src="https://github.com/user-attachments/assets/e08cad8a-ff8c-4336-b34b-da92ef959a96" />
  <br>
  <em>Comparison of different dimensions of Latent Transition (Reconstruction loss only)</em>
</p>




* **Full VAE (Reconstruction + KL):** The transitions are visually smooth, gradual, and semantically coherent. For instance, when morphing from a '6' to a '0', or a '1' to a '0', the digit steadily alters its geometry. It widens smoothly, curves at its outer boundaries, develops rounded lobes, and establishes plausible intermediate semi-open or semi-closed digit-like forms before settling into a clean oval '0'. This behavior confirms that KL regularization ensures a well-behaved, continuous manifold where empty space maps to realistic data patterns.

* **Reconstruction-Only VAE:** While a 2D space yields a relatively clean visual shift, higher dimensions (5 to 100) expose severe structural instability. The transitions are highly unorganized and erratic. Instead of a gradual morph, the intermediate stages display sudden morphological jumps, abrupt structural collapses into irregular blob-like states, or heavily blurred and noisy forms before abruptly snaps into the target digit. This highlights that omitting the KL term creates unconstrained voids and sharp geometric rifts in the latent space.

* **KL-Only VAE:** The transition fails completely across all timesteps and all dimensions. Because the decoder never learned to map latent coordinates back to meaningful image structures, every step along the interpolation trajectory yields static, uniform white noise.

---

### 3 Comparative Latent Space Analysis Framework

The architectural paradigms of the VAE and the Diffusion Model govern how they represent, organize, and interpret data internally. This performance profile evaluates these differences across three primary dimensions:

<p align="center">
  <img width="605" height="484" alt="image" src="https://github.com/user-attachments/assets/351b9dbf-b6b4-406c-a37e-0b808343d964" />
  <br>
    <em>The image generated from VAE and Diffusion mode (Top: VAE, Bottom: Diffusion model)</em>
  <img width="605" height="176" alt="image" src="https://github.com/user-attachments/assets/63c31e0c-ee45-4b43-b756-25d18456ac5b" />
  <br>
    <em>The sampling trajectory of a diffusion model</em>
</p>

#### 1. Dimensionality & Structural Setup

* **VAE:** Relies on an explicit, low-dimensional spatial bottleneck. Data is compressed into fixed spatial coordinates where different digit classes form static clusters. Due to the trade-off between the Gaussian prior and reconstruction constraints, these clusters frequently overlap at their boundaries.
* **Diffusion Model:** Lacks a static, low-dimensional bottleneck. Its latent structure is high-dimensional and implicit, mapped across a temporal sequence of noisy images over $T = 1000$ reverse diffusion steps (Figure 8). Starting from pure noise, the model iteratively moves toward the real data distribution, achieving a much cleaner separation of the data distribution than the VAE's overlapping spatial layout.

#### 2. Semantic Interpretability
* **VAE:** Highly interpretable in a direct visual sense. Plotting the 2D spatial plane with color-coded class labels reveals clear class locations and allows users to track exactly how features shift across the coordinate space during interpolation.
* **Diffusion Model:** Low direct structural interpretability from a single static representation. The generation mechanism remains hidden within a complex noise schedule and can only be evaluated by visualizing the full reverse denoising path over time.

#### 3. Feature Disentanglement
* **VAE:** Disentanglement is structurally limited. Because class clusters overlap heavily in the lower-dimensional plane, features from different digit classes often blend together at the boundaries.
* **Diffusion Model:** Displays naturally emergent feature disentanglement over its reverse execution path. Global structural features appear during the early denoising stages, while fine-grained, class-specific details isolate cleanly in the final steps, preventing the attribute mixing common in the VAE bottleneck.

---

### 4 Quantitative Evaluation of Generative Quality and Utility

To rigorously quantify the synthesis quality and downstream utility of the generated images, all models were evaluated using the Fréchet Inception Distance (FID) and a synthetic data training classification suite.

### Table 1: Statistical Evaluation of Generative Performance

| Evaluation Dataset Source | Fréchet Inception Distance (FID) ↓ | Downstream Classifier Accuracy ↑ |
| --- | --- | --- |
| **VAE-Generated Dataset** | 121.11 | 59.93% |
| **Diffusion Model (DM) Dataset** | 34.25 | 77.31% |
| **Original MNIST Dataset (Baseline)** | — | 99.21% |

> **Note on Metrics:** A lower FID score indicates closer alignment with the real data distribution. Higher downstream classifier accuracy indicates that the synthetic data preserves more realistic, discriminative features.
> 
> 

#### FID Performance Analysis

<p align="center">
  <img width="499" height="124" alt="image" src="https://github.com/user-attachments/assets/09f43ec0-0d27-4df3-aca8-a2c4e5223760" />
  <img width="499" height="124" alt="image" src="https://github.com/user-attachments/assets/09f43ec0-0d27-4df3-aca8-a2c4e5223760" />
  <br>
  <em>The image generated from VAE and Diffusion mode (Top: VAE, Bottom: Diffusion model)</em>
</p>

The Diffusion Model significantly outperforms the VAE, achieving an FID score of **34.25** compared to the VAE's **121.11**. This performance gap stems from the VAE’s low-dimensional information bottleneck, which forces the encoder to drop high-frequency details during compression. As a result, the VAE decoder outputs blurry, smoothed images that lack fine details. Furthermore, feature bleeding at overlapping cluster boundaries causes the VAE to occasionally generate hybrid digits containing traits from multiple classes, inflating its FID score. In contrast, the Diffusion Model's iterative refinement process preserves sharp edge details and structural clarity.

#### Downstream Task Utility Analysis

The utility of the synthetic data was tested by training separate classifiers on 50,000 generated images from each model and evaluating them on the real MNIST test set.

* The classifier trained on VAE data achieved an accuracy of **59.93%**, confirming that VAE samples retain only coarse, class-level shapes while losing the sharp variations needed for robust classification.


* The classifier trained on Diffusion Model data reached an accuracy of **77.31%**, proving that the diffusion framework captures the real data manifold with much higher fidelity.


* Both models still lag behind the **99.21%** baseline achieved using real training data. This remaining gap is caused by subtle pixel-level discrepancies in the synthetic images that act as mild adversarial distortions, confounding downstream neural networks trained entirely on generated data.



---

### 5 Computational Efficiency & Sampling Speed Profiles

Image generation throughput was measured under identical hardware configurations to quantify the operational cost of each architecture.

### Generative Sampling Speed Profiling

| Generative Model Architecture | Average Generation Time per Image (Seconds) |
| --- | --- |
| **Variational Autoencoder (VAE)** | **4 ± 0.5** seconds |
| **Denoising Diffusion Probabilistic Model (DDPM)** | **90 ± 10** seconds |

#### Efficiency Trade-off Analysis

The runtime profiles highlight a clear trade-off between generative speed and output quality. The VAE offers exceptionally fast sampling, requiring only **4 ± 0.5** seconds per image. Because it uses a single-step generation mechanism, a sampled latent vector $z$ is converted into a completed output image through a single parallelized forward pass through the decoder network.

Conversely, the Diffusion Model is more than twenty times slower, requiring **90 ± 10** seconds per image. This bottleneck occurs because the model must execute a sequential loop of $T = 1000$ reverse denoising steps. Each individual step requires a full forward pass through the deep U-Net architecture to predict and subtract noise. Because the input for step $t$ depends strictly on the output of step $t-1$, these operations cannot be parallelized, creating a significant real-time computational constraint.

---

### 6 Industrial Anomaly Detection (IAD) Pipeline Application Results

<p align="center">
  <img width="1015" height="392" alt="image" src="https://github.com/user-attachments/assets/9d914ebb-b9fd-4921-9e2b-b5f3147ef776" />
  <br>
  <em>Anomaly detection result on glass bottle</em>
</p>


To test practical utility beyond standard academic benchmarks, a reconstruction-based anomaly detection pipeline was evaluated on the glass bottle category of the industrial MVTec dataset.

#### Defect Detection & Heatmap Localization Mechanics

Because the underlying U-Net architecture was trained exclusively on defect-free, nominal glass bottle samples, it learns to map only normal structural distributions. When an anomalous image containing a physical defect is introduced, the model cannot reconstruct the out-of-distribution anomaly because it lies completely outside the learned manifold.

During inference, the absolute pixel-wise difference map between the original input and the clean reconstruction spikes sharply at the defect location. This error map is successfully translated into a spatial anomaly heatmap, allowing the system to isolate subtle anomalies and identify defects with high precision. Images are then classified by comparing the aggregate anomaly score against a fixed threshold (e.g., classifying a sample as an anomaly with a score of **0.0457** against a threshold of **0.0150**).

## Hardware Constraints and Project Limitations
While all project goals were achieved, specific computational limitations must be noted:
* Due to restricted access to high-performance enterprise GPUs, the project utilized relatively shallow networks, limited training epochs, and low-resolution inputs.
* The DDPM's U-Net utilized fewer channels and attention blocks than standard state-of-the-art networks, and a simplified architecture was deployed for industrial inspection due to VRAM limitations. Consequently, the reported metrics (FID and Accuracy) represent conservative lower bounds of what these frameworks can achieve with full computational scale.

---

## References
1.	Chen, R. T. Q. (2023). TorchCFM: Conditional flow matching in PyTorch . arXiv. https://arxiv.org/abs/2302.00482 
2.	Chen, R. T. Q. (2024). Flow matching for generative modeling. In The Proceedings of the 41st International Conference on Machine Learning (ICML).
3.	Higgins, I., Loic Matthey, Arka Pal, Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., & Lerchner, A. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. Openreview.net. https://openreview.net/forum?id=Sy2fzU9gl.
4.	Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, 33, 6840–6851.
5.	Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based generative models. In Advances in Neural Information Processing Systems, 35, 26565–26577.
6.	Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. In 2nd International Conference on Learning Representations (ICLR).
7.	Kingma, D. P., & Welling, M. (2019). An introduction to variational autoencoders. Foundations and Trends in Machine Learning, 12(4), 307–392.
8.	Kingma, D. P., Salimans, T., Poole, B., & Ho, J. (2021). Variational diffusion models. In Advances in Neural Information Processing Systems, 34, 21696–21707.
9.	Nichol, A. Q., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. In Proceedings of the 38th International Conference on Machine Learning (ICML), 8162–8171.
10.	Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P. (2021, June 15). Towards total recall in industrial anomaly detection. arXiv.org. https://arxiv.org/abs/2106.08265
11.	Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 10684–10695.
12.	Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings of the 32nd International Conference on Machine Learning (ICML), 37, 2256–2265.
13.	Song, Y., & Ermon, S. (2021). Generative modeling by estimating gradients of the data distribution. In Advances in Neural Information Processing Systems, 34, 11918–11930. 
14.	Song, J., Meng, C., & Ermon, S. (2022). Denoising diffusion implicit models. In 9th International Conference on Learning Representations (ICLR). 
15.	Sordo, Z., Chagnon, E., Hu, Z., Donatelli, J. J., Andeer, P., Nico, P. S., Northen, T., & Ushizima, D. (2025). Synthetic Scientific Image Generation with VAE, GAN, and Diffusion Model Architectures. Journal of imaging, 11(8), 252. https://doi.org/10.3390/jimaging11080252 
16.	Van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural discrete representation learning. In Advances in Neural Information Processing Systems, 30, 6306–6315.
17.	Zhao, S., Song, J., & Ermon, S. (2019). Infovae: Balancing learning and inference in variational autoencoders. In Proceedings of the AAAI Conference on Artificial Intelligence, 33, 5885–5892.
"""


