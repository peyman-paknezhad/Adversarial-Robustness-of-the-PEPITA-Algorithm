This code implements the PEPITA algorithm, which is detailed in the paper "Intrinsic Biologically Plausible Adversarial Robustness." PEPITA offers a biologically-inspired alternative to Backpropagation (BP) and emphasizes adversarial robustness. Some aspects were inspired by the repository at https://github.com/GiorgiaD/PEPITA.

### PEPITA Algorithm Explanation

PEPITA is a biologically-inspired learning algorithm that serves as an alternative to Backpropagation (BP). It avoids the need for a separate backward pass to compute gradients. Below is a step-by-step breakdown of PEPITA based on the provided text.

#### 1. First Forward Pass (Standard Forward Pass)
The algorithm begins with a typical forward pass, identical to BP.

- Input $( x )$ is passed through the network layer by layer. The activation of the $( i )$-th layer is:
  $`
  h_i = \sigma_i(W_i h_{i-1})
  `$
  
- The final output $( h_L )$ is compared to the target output $( y^* )$, and the error signal $( e )$ is computed as:
  $`
  e = h_L - y^*
  `$

#### 2. Second Forward Pass (Modulated Forward Pass)
Instead of backpropagating the error signal, PEPITA adds the error $( e )$ to the original input $( x )$ via a fixed random feedback projection matrix $( F )$.

- The modulated input becomes $( x_{\text{mod}} = x + Fe )$, which is used for a second forward pass through the network.
  
- For each layer, the modulated activation is computed as:
  $`
  h_{\text{mod}}^i = \sigma_i(W_i h_{\text{mod}}^{i-1})
  `$

#### 3. Weight Update Based on Differences in Activations
The weight updates in PEPITA are computed by comparing neuron activations from the first forward pass and the second (modulated) forward pass.

- For the hidden layers $( i < L )$, the weight update is given by:
  $`
  \Delta W_i = (h_i - h_{\text{mod}}^i) \cdot (h_{\text{mod}}^{i-1})^T
  `$
  
- For the output layer, the weight update uses the error signal $( e )$ directly:
  $`
  \Delta W_L = e \cdot (h_{\text{mod}}^{L-1})^T
  `$

#### 4. Weight Updates Comparison with BP
In BP, the weight updates for the output layer are computed using the gradient of the loss with respect to the weights:

$`
\Delta W_L^{BP} = e \cdot \sigma_L'(z_L) \cdot (h_{L-1})^T
`$
where $( z_L = W_L h_{L-1} )$.

In PEPITA, for the output layer, the weight update equation simplifies to:
$`
\Delta W_L^{PEPITA} = e \cdot (h_{\text{mod}}^{L-1})^T
`$
This closely approximates BP, as the perturbations introduced by the feedback projection matrix $( F )$ are small.

#### 5. Hidden Layer Weight Updates
For hidden layers, the BP weight update is recursively calculated using the error backpropagation formula:
$`
\Delta W_1^{BP} = \delta_1 \cdot x^T
`$
where $( \delta_1 = (W_2^T \delta_2) \cdot \sigma_1'(z_1) )$ and $( \delta_2 = e )$.

In contrast, the PEPITA weight update for the hidden layer is:
$`
\Delta W_1^{PEPITA} = (h_1 - h_{\text{mod}}^1) \cdot (x_{\text{mod}})^T
`$
This shows a difference in how the updates are derived compared to BP but still leverages local information from the difference in activations.


### Experiments 

The paper presents three key experiments:

#### 1. **Baseline Natural and Adversarial Performance**
   - The models trained with PEPITA and BP were compared on MNIST
   - Results show that while PEPITA's natural accuracy is slightly lower than BP's, neither model is robust to adversarial attacks without adversarial training.

#### 2. **Intrinsic Adversarial Robustness**
   - When selecting hyperparameters based on adversarial validation accuracy, PEPITA exhibited significantly better adversarial robustness than BP.
   - PEPITA’s adversarial accuracy was much higher with a smaller drop in natural accuracy compared to BP, which showed a more severe natural-vs-adversarial performance trade-off.

#### 3. **Adversarial Training**
   - Both models were adversarially trained using PGD adversarial samples.
   - PEPITA demonstrated better adversarial test accuracy and less degradation in natural performance, showcasing a better natural-vs-adversarial trade-off than BP.

#### 4. Fast Adversarial Training

  - Both PEPITA and BP were adversarially trained using FGSM adversarial samples, with evaluation using the stronger PGD attack.
  - Results show that PEPITA maintained a much higher adversarial accuracy compared to BP.
  - Additionally, PEPITA showed better generalization from FGSM samples, with less degradation in adversarial performance, making it more efficient for fast adversarial training without suffering from catastrophic overfitting.
### Results

- **PEPITA** achieved a higher intrinsic adversarial robustness than BP, maintaining strong performance on adversarial tasks while minimizing natural performance loss.
- **BP** suffered from a larger drop in adversarial accuracy, particularly when facing strong adversarial attacks like PGD.
- When using fast adversarial training (with FGSM samples), PEPITA generalized better to stronger adversarial attacks than BP.

Overall, PEPITA significantly outperformed BP in terms of adversarial robustness, offering a more biologically plausible and computationally efficient alternative.


