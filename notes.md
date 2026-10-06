LLaDA is a Diffusio, model trained from scratch under the pre-trained and **selfsupervised fine-tuning** paradigm. 

we have to employ a data masking process and a reverse generation process, parameterized by transformer to predict masked tokens. 

It is good for reversal curse, like completing a the begining of a sentence. So it might be interesting to test our model on these king of tasks for the platform.

The key insight is that llada will use the generative modeling principles rather than the autoregressive formulation. 

LLaDA leverages a masked diffusion model (MDM), which incorporates a forward data masking process and trains a *mask predictor* to approximate the reverse process. The design enables LLaDA to construct a model distribution with bidrectional dependancies and optimize a variational lower bound of it's log-likelihood.

The original paper adopt a standart piples of data preparation, pre-training, SFT and evaluation. 

LLaDA defins a model destribution $p_\theta(x)$ trought a forward process and a reverse process. During forward we gradually mask tokens independatly in $x_0$ until the sequence is fully masked at $t=1$. For $t\in (0,1)$, the sequence $x_t$ is partially masked. The reverse process recovers the data distribution by iterativly predicting masked tokens as t moves from 1 to 0. 

the code of LLaDA is a masked predictor, a parametric model $p_\theta(·|x_t)$ that takes $x_t$ as input and predicts masked tokens (M) simultaneously.

It is trained on Cross-Entropy Loss computed only on the masked tokens :
    $$\mathcal{L}(\theta) = -\mathbb{E}_{t, x_0, x_t} \left[ \frac{1}{t} \sum_{i \in L} \mathbf{1}_{[x_{t}^i = M]} \log p_\theta(x_0^i | x_t) \right]$$
where $L$ is the sequence length, $x_0$ is a training sample, $t$ is a continuous random variable drawn uniformly from [0, 1], and $x_t$ is sampled from the forward process. The indicator function ensures that the loss is computed only for maksed tokens. 

Once we train the model, we can simulate a reverse process parametrized by the mask predictor and define the model distribution $p_\theta(x_0)$ as the marginal distribution of $x_0$ induced at $t=0.

LlaDa uses a making ratio that is random from 0 to 1. 

## Pre-training

LLaDA uses Tranformer as amaks predictor. 

They used vanilla multi-head attention insead of group query attention for simplicity. Consiquntly, th attention layer has more parameters and they reduce the FFN dimension to maintain a similar number of parameters as LLaMA. I am not sure if we have to do the same for our model, as we don't want to compare to LLaMA but just a model that works well. 

They addopt the Warmup-Stable-decay lr schedular to monitor trainig progress. They lineary increased the learning rate from 0 to 4e-4 over first 2000 iterations and maintained it at 4e-4. Fater tprocessing 1.2T tokens they decay it 1e-4 and etc. They used AdamW optimizer with weight decay of 0.1.

## Superised Fine-Tuning

They enhanced the capability of LLaDA to follow instructions by SFT with paired data $(p_0, r_0)$, where $p_0$ is the prompt and $r_0$ denotes the reponse. This requires $p_\theta(r_0|p_0)$ instead of $p_\theta(x_0)$. 

The implementation is simmilar to pre-training. They leave the prompt unchanged and mask the tokens in the response independently as done for $x_0$. Then, they feed both the prompt and masked response $r_t$ to the pre-trained mask predictor to compute the loss for SFT:
    $$-\mathbb{E}_{t, p_0, r_0, r_t} \left[ \frac{1}{t} \sum_{i \in L'} \mathbf{1}_{[r_{t}^i = M]} \log p_\theta(r_0^i | p_0, r_t) \right]$$
where L' denotes a dynamic length specifed later. 


