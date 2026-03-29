<div align="center">
<h1>CounterScene: Counterfactual Causal Reasoning in Generative World Models for Safety-Critical Closed-Loop Evaluation</h1>

<p>
Bowen Jing<sup>1,*</sup>, Ruiyang Hao<sup>2,*</sup>, Weitao Zhou<sup>3,&Dagger;</sup>, Haibao Yu<sup>1,4,&Dagger;</sup>
</p>

<p>
<sup>1</sup> Tuojing Intelligence, <sup>2</sup> King's College London, <br>
<sup>3</sup> Tsinghua University, <sup>4</sup> The University of Hong Kong
</p>

<p>
<sup>*</sup> Equal contribution &nbsp; <sup>&Dagger;</sup> Corresponding authors
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2603.21104-b31b1b)](https://arxiv.org/abs/2603.21104)&nbsp;
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red)](https://arxiv.org/pdf/2603.21104)&nbsp;
[![Release](https://img.shields.io/badge/Release-Coming%20Soon-blue)](#release-status)

</div>

## News

- **`Mar. 2026`:** CounterScene is available on [arXiv](https://arxiv.org/abs/2603.21104).
- **`Coming soon`:** We will release the full codebase, trained checkpoints, and evaluation scripts.

## Table of Contents

- [Introduction](#introduction)
- [Method Overview](#method-overview)
- [Main Results](#main-results)
- [Qualitative Results](#qualitative-results)
- [Release Status](#release-status)
- [Citation](#citation)

## Introduction

CounterScene studies safety-critical scenario generation through **counterfactual causal reasoning** in generative world models.

Rather than applying arbitrary perturbations, CounterScene identifies the agent whose behavior is causally maintaining safety and performs a **minimal intervention** that transforms a safe scene into a realistic safety-critical interaction while preserving coherent multi-agent dynamics.

<div align="center"><b>Overview of CounterScene.</b>
<img src="assets/intro.jpg" />
<p>CounterScene turns an observed safe traffic scene into a realistic safety-critical interaction by intervening on the causally critical agent while preserving the consistency of the surrounding world.</p>
</div>
<br>

## Method Overview

CounterScene consists of four key components:

- **Causal adversarial agent selection** identifies the agent whose current behavior suppresses the greatest latent risk.
- **Causal Interaction Graph (CIG)** models conflict-aware dependencies among agents.
- **Counterfactual guidance** perturbs only the selected agent trajectory during diffusion.
- **Closed-loop rollout** allows all other agents to react naturally, preserving realistic multi-agent dynamics.

<div align="center"><b>Framework of CounterScene.</b>
<img src="assets/methodology.jpg" />
</div>

## Main Results

CounterScene achieves a stronger realism-effectiveness trade-off than prior baselines on **nuScenes**. To provide a concise comparison against all major baselines, we report representative results at the **5s horizon** below.

| Horizon | Method | ADE | FDE | OOR | HBR | CR |
| :-- | :-- | --: | --: | --: | --: | --: |
| 5s | CTG | 1.245 | 2.977 | 0.3% | 1.6% | 1.0% |
| 5s | VAE | 1.497 | 3.597 | 0.5% | 0.2% | 6.0% |
| 5s | STRIVE | 1.215 | 3.078 | 0.5% | **0.1%** | 6.0% |
| 5s | CTG++ | 1.273 | 3.190 | **0.2%** | 1.1% | 1.0% |
| 5s | CCDiff | 0.924 | 2.421 | 1.2% | 1.5% | 3.0% |
| 5s | **CounterScene** | **0.731** | **1.967** | 1.1% | 2.0% | **11.0%** |

**Metric note.** ADE/FDE measure trajectory realism, while OOR, HBR, and CR evaluate off-road rate, hard-brake rate, and collision rate.

For full per-horizon comparisons, qualitative results, ablations, and zero-shot transfer results, please refer to the paper.

### Full Per-Horizon Results

<details>
<summary>Click to expand the full per-horizon results on nuScenes (10 Hz)</summary>

This section reports full per-horizon results on **nuScenes (10 Hz)**. Lower is better for `ADE`, `FDE`, `OOR`, and `HBR`, while higher `CR` indicates stronger safety-critical scenario induction.

<table>
  <thead>
    <tr>
      <th>Horizon</th>
      <th>Metric</th>
      <th>CTG</th>
      <th>VAE</th>
      <th>STRIVE</th>
      <th>CTG++</th>
      <th>CCDiff</th>
      <th>CounterScene</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5"><strong>1s</strong></td>
      <td>ADE &darr;</td>
      <td align="right">0.178</td>
      <td align="right">0.206</td>
      <td align="right">0.170</td>
      <td align="right">0.185</td>
      <td align="right">0.125</td>
      <td align="right"><strong>0.089</strong></td>
    </tr>
    <tr>
      <td>FDE &darr;</td>
      <td align="right">0.348</td>
      <td align="right">0.415</td>
      <td align="right">0.326</td>
      <td align="right">0.361</td>
      <td align="right">0.247</td>
      <td align="right"><strong>0.174</strong></td>
    </tr>
    <tr>
      <td>OOR &darr;</td>
      <td align="right">0.2%</td>
      <td align="right">0.3%</td>
      <td align="right">0.2%</td>
      <td align="right">0.2%</td>
      <td align="right">0.2%</td>
      <td align="right">0.2%</td>
    </tr>
    <tr>
      <td>HBR &darr;</td>
      <td align="right">1.9%</td>
      <td align="right">0.5%</td>
      <td align="right"><strong>0.2%</strong></td>
      <td align="right">1.5%</td>
      <td align="right">1.4%</td>
      <td align="right">1.3%</td>
    </tr>
    <tr>
      <td>CR &uarr;</td>
      <td align="right">0.0%</td>
      <td align="right">0.0%</td>
      <td align="right">0.0%</td>
      <td align="right">0.0%</td>
      <td align="right">0.0%</td>
      <td align="right">0.0%</td>
    </tr>
    <tr>
      <td rowspan="5"><strong>2s</strong></td>
      <td>ADE &darr;</td>
      <td align="right">0.393</td>
      <td align="right">0.467</td>
      <td align="right">0.359</td>
      <td align="right">0.409</td>
      <td align="right">0.263</td>
      <td align="right"><strong>0.184</strong></td>
    </tr>
    <tr>
      <td>FDE &darr;</td>
      <td align="right">0.854</td>
      <td align="right">1.037</td>
      <td align="right">0.783</td>
      <td align="right">0.902</td>
      <td align="right">0.589</td>
      <td align="right"><strong>0.410</strong></td>
    </tr>
    <tr>
      <td>OOR &darr;</td>
      <td align="right"><strong>0.2%</strong></td>
      <td align="right">0.4%</td>
      <td align="right">0.3%</td>
      <td align="right"><strong>0.2%</strong></td>
      <td align="right">0.3%</td>
      <td align="right">0.3%</td>
    </tr>
    <tr>
      <td>HBR &darr;</td>
      <td align="right">1.7%</td>
      <td align="right">0.4%</td>
      <td align="right"><strong>0.0%</strong></td>
      <td align="right">1.3%</td>
      <td align="right">1.5%</td>
      <td align="right">1.6%</td>
    </tr>
    <tr>
      <td>CR &uarr;</td>
      <td align="right">0.0%</td>
      <td align="right">0.0%</td>
      <td align="right">0.0%</td>
      <td align="right">0.0%</td>
      <td align="right">1.0%</td>
      <td align="right"><strong>3.0%</strong></td>
    </tr>
    <tr>
      <td rowspan="5"><strong>3s</strong></td>
      <td>ADE &darr;</td>
      <td align="right">0.634</td>
      <td align="right">0.770</td>
      <td align="right">0.600</td>
      <td align="right">0.692</td>
      <td align="right">0.437</td>
      <td align="right"><strong>0.338</strong></td>
    </tr>
    <tr>
      <td>FDE &darr;</td>
      <td align="right">1.425</td>
      <td align="right">1.773</td>
      <td align="right">1.410</td>
      <td align="right">1.589</td>
      <td align="right">1.033</td>
      <td align="right"><strong>0.814</strong></td>
    </tr>
    <tr>
      <td>OOR &darr;</td>
      <td align="right"><strong>0.2%</strong></td>
      <td align="right">0.4%</td>
      <td align="right">0.4%</td>
      <td align="right"><strong>0.2%</strong></td>
      <td align="right">0.5%</td>
      <td align="right">0.5%</td>
    </tr>
    <tr>
      <td>HBR &darr;</td>
      <td align="right">1.5%</td>
      <td align="right">0.3%</td>
      <td align="right"><strong>0.1%</strong></td>
      <td align="right">1.0%</td>
      <td align="right">1.5%</td>
      <td align="right">1.7%</td>
    </tr>
    <tr>
      <td>CR &uarr;</td>
      <td align="right">0.0%</td>
      <td align="right">0.0%</td>
      <td align="right">1.0%</td>
      <td align="right">0.0%</td>
      <td align="right">2.0%</td>
      <td align="right"><strong>4.0%</strong></td>
    </tr>
    <tr>
      <td rowspan="5"><strong>4s</strong></td>
      <td>ADE &darr;</td>
      <td align="right">0.900</td>
      <td align="right">1.107</td>
      <td align="right">0.880</td>
      <td align="right">0.906</td>
      <td align="right">0.695</td>
      <td align="right"><strong>0.542</strong></td>
    </tr>
    <tr>
      <td>FDE &darr;</td>
      <td align="right">2.102</td>
      <td align="right">2.605</td>
      <td align="right">2.166</td>
      <td align="right">2.183</td>
      <td align="right">1.764</td>
      <td align="right"><strong>1.414</strong></td>
    </tr>
    <tr>
      <td>OOR &darr;</td>
      <td align="right"><strong>0.2%</strong></td>
      <td align="right">0.4%</td>
      <td align="right">0.4%</td>
      <td align="right"><strong>0.2%</strong></td>
      <td align="right">0.6%</td>
      <td align="right">0.8%</td>
    </tr>
    <tr>
      <td>HBR &darr;</td>
      <td align="right">1.5%</td>
      <td align="right">0.2%</td>
      <td align="right"><strong>0.1%</strong></td>
      <td align="right">1.2%</td>
      <td align="right">1.4%</td>
      <td align="right">1.9%</td>
    </tr>
    <tr>
      <td>CR &uarr;</td>
      <td align="right">0.0%</td>
      <td align="right">5.0%</td>
      <td align="right">4.0%</td>
      <td align="right">3.0%</td>
      <td align="right">2.0%</td>
      <td align="right"><strong>6.0%</strong></td>
    </tr>
    <tr>
      <td rowspan="5"><strong>5s</strong></td>
      <td>ADE &darr;</td>
      <td align="right">1.245</td>
      <td align="right">1.497</td>
      <td align="right">1.215</td>
      <td align="right">1.273</td>
      <td align="right">0.924</td>
      <td align="right"><strong>0.731</strong></td>
    </tr>
    <tr>
      <td>FDE &darr;</td>
      <td align="right">2.977</td>
      <td align="right">3.597</td>
      <td align="right">3.078</td>
      <td align="right">3.190</td>
      <td align="right">2.421</td>
      <td align="right"><strong>1.967</strong></td>
    </tr>
    <tr>
      <td>OOR &darr;</td>
      <td align="right">0.3%</td>
      <td align="right">0.5%</td>
      <td align="right">0.5%</td>
      <td align="right"><strong>0.2%</strong></td>
      <td align="right">1.2%</td>
      <td align="right">1.1%</td>
    </tr>
    <tr>
      <td>HBR &darr;</td>
      <td align="right">1.6%</td>
      <td align="right">0.2%</td>
      <td align="right"><strong>0.1%</strong></td>
      <td align="right">1.1%</td>
      <td align="right">1.5%</td>
      <td align="right">2.0%</td>
    </tr>
    <tr>
      <td>CR &uarr;</td>
      <td align="right">1.0%</td>
      <td align="right">6.0%</td>
      <td align="right">6.0%</td>
      <td align="right">1.0%</td>
      <td align="right">3.0%</td>
      <td align="right"><strong>11.0%</strong></td>
    </tr>
    <tr>
      <td rowspan="5"><strong>6s</strong></td>
      <td>ADE &darr;</td>
      <td align="right">1.503</td>
      <td align="right">1.873</td>
      <td align="right">1.558</td>
      <td align="right">1.767</td>
      <td align="right">1.205</td>
      <td align="right"><strong>1.059</strong></td>
    </tr>
    <tr>
      <td>FDE &darr;</td>
      <td align="right">3.620</td>
      <td align="right">4.528</td>
      <td align="right">4.001</td>
      <td align="right">4.438</td>
      <td align="right">3.179</td>
      <td align="right"><strong>2.810</strong></td>
    </tr>
    <tr>
      <td>OOR &darr;</td>
      <td align="right"><strong>0.2%</strong></td>
      <td align="right">0.6%</td>
      <td align="right">0.5%</td>
      <td align="right">0.3%</td>
      <td align="right">1.4%</td>
      <td align="right">1.5%</td>
    </tr>
    <tr>
      <td>HBR &darr;</td>
      <td align="right">1.5%</td>
      <td align="right">0.2%</td>
      <td align="right"><strong>0.1%</strong></td>
      <td align="right">1.3%</td>
      <td align="right">1.7%</td>
      <td align="right">2.0%</td>
    </tr>
    <tr>
      <td>CR &uarr;</td>
      <td align="right">2.0%</td>
      <td align="right">7.0%</td>
      <td align="right">7.0%</td>
      <td align="right">3.0%</td>
      <td align="right">5.0%</td>
      <td align="right"><strong>15.0%</strong></td>
    </tr>
    <tr>
      <td rowspan="5"><strong>7s</strong></td>
      <td>ADE &darr;</td>
      <td align="right">1.910</td>
      <td align="right">2.291</td>
      <td align="right">1.921</td>
      <td align="right">2.149</td>
      <td align="right">1.255</td>
      <td align="right"><strong>1.155</strong></td>
    </tr>
    <tr>
      <td>FDE &darr;</td>
      <td align="right">4.708</td>
      <td align="right">5.525</td>
      <td align="right">4.974</td>
      <td align="right">5.620</td>
      <td align="right"><strong>3.141</strong></td>
      <td align="right"><strong>3.141</strong></td>
    </tr>
    <tr>
      <td>OOR &darr;</td>
      <td align="right">0.3%</td>
      <td align="right">0.7%</td>
      <td align="right">0.6%</td>
      <td align="right">0.3%</td>
      <td align="right">1.4%</td>
      <td align="right">1.4%</td>
    </tr>
    <tr>
      <td>HBR &darr;</td>
      <td align="right">1.4%</td>
      <td align="right">0.2%</td>
      <td align="right"><strong>0.1%</strong></td>
      <td align="right">1.2%</td>
      <td align="right">1.3%</td>
      <td align="right">1.9%</td>
    </tr>
    <tr>
      <td>CR &uarr;</td>
      <td align="right">4.0%</td>
      <td align="right">10.0%</td>
      <td align="right">10.0%</td>
      <td align="right">3.0%</td>
      <td align="right"><strong>18.0%</strong></td>
      <td align="right"><strong>18.0%</strong></td>
    </tr>
    <tr>
      <td rowspan="5"><strong>8s</strong></td>
      <td>ADE &darr;</td>
      <td align="right">2.149</td>
      <td align="right">2.667</td>
      <td align="right">2.303</td>
      <td align="right">2.632</td>
      <td align="right">1.726</td>
      <td align="right"><strong>1.551</strong></td>
    </tr>
    <tr>
      <td>FDE &darr;</td>
      <td align="right">5.274</td>
      <td align="right">6.413</td>
      <td align="right">5.973</td>
      <td align="right">6.699</td>
      <td align="right">4.606</td>
      <td align="right"><strong>4.184</strong></td>
    </tr>
    <tr>
      <td>OOR &darr;</td>
      <td align="right"><strong>0.2%</strong></td>
      <td align="right">0.8%</td>
      <td align="right">0.7%</td>
      <td align="right">0.3%</td>
      <td align="right">1.8%</td>
      <td align="right">1.5%</td>
    </tr>
    <tr>
      <td>HBR &darr;</td>
      <td align="right">1.4%</td>
      <td align="right">0.2%</td>
      <td align="right"><strong>0.0%</strong></td>
      <td align="right">1.3%</td>
      <td align="right">1.4%</td>
      <td align="right">1.9%</td>
    </tr>
    <tr>
      <td>CR &uarr;</td>
      <td align="right">0.0%</td>
      <td align="right">11.0%</td>
      <td align="right">14.0%</td>
      <td align="right">2.0%</td>
      <td align="right">9.0%</td>
      <td align="right"><strong>18.0%</strong></td>
    </tr>
    <tr>
      <td rowspan="5"><strong>9s</strong></td>
      <td>ADE &darr;</td>
      <td align="right">2.571</td>
      <td align="right">3.082</td>
      <td align="right">2.737</td>
      <td align="right">2.925</td>
      <td align="right">2.184</td>
      <td align="right"><strong>1.812</strong></td>
    </tr>
    <tr>
      <td>FDE &darr;</td>
      <td align="right">6.363</td>
      <td align="right">7.380</td>
      <td align="right">7.087</td>
      <td align="right">7.483</td>
      <td align="right">5.835</td>
      <td align="right"><strong>5.087</strong></td>
    </tr>
    <tr>
      <td>OOR &darr;</td>
      <td align="right"><strong>0.2%</strong></td>
      <td align="right">0.9%</td>
      <td align="right">0.8%</td>
      <td align="right"><strong>0.2%</strong></td>
      <td align="right">2.4%</td>
      <td align="right">1.9%</td>
    </tr>
    <tr>
      <td>HBR &darr;</td>
      <td align="right">1.5%</td>
      <td align="right">0.2%</td>
      <td align="right"><strong>0.1%</strong></td>
      <td align="right">1.2%</td>
      <td align="right">1.6%</td>
      <td align="right">1.8%</td>
    </tr>
    <tr>
      <td>CR &uarr;</td>
      <td align="right">3.0%</td>
      <td align="right">14.0%</td>
      <td align="right">15.0%</td>
      <td align="right">5.0%</td>
      <td align="right">12.0%</td>
      <td align="right"><strong>24.0%</strong></td>
    </tr>
    <tr>
      <td rowspan="5"><strong>10s</strong></td>
      <td>ADE &darr;</td>
      <td align="right">2.720</td>
      <td align="right">3.509</td>
      <td align="right">3.126</td>
      <td align="right">3.333</td>
      <td align="right">2.367</td>
      <td align="right"><strong>2.267</strong></td>
    </tr>
    <tr>
      <td>FDE &darr;</td>
      <td align="right">6.792</td>
      <td align="right">8.505</td>
      <td align="right">8.119</td>
      <td align="right">8.392</td>
      <td align="right">7.253</td>
      <td align="right"><strong>6.153</strong></td>
    </tr>
    <tr>
      <td>OOR &darr;</td>
      <td align="right"><strong>0.2%</strong></td>
      <td align="right">1.2%</td>
      <td align="right">0.9%</td>
      <td align="right"><strong>0.2%</strong></td>
      <td align="right">2.6%</td>
      <td align="right">2.3%</td>
    </tr>
    <tr>
      <td>HBR &darr;</td>
      <td align="right">1.4%</td>
      <td align="right">0.2%</td>
      <td align="right"><strong>0.1%</strong></td>
      <td align="right">1.4%</td>
      <td align="right">1.4%</td>
      <td align="right">1.7%</td>
    </tr>
    <tr>
      <td>CR &uarr;</td>
      <td align="right">3.0%</td>
      <td align="right">15.0%</td>
      <td align="right">17.0%</td>
      <td align="right">4.0%</td>
      <td align="right">16.0%</td>
      <td align="right"><strong>26.0%</strong></td>
    </tr>
  </tbody>
</table>

</details>

## Qualitative Results

### Scene 103: Merging Cut-in

<div align="center">
<img src="assets/103.png" width="900" />
<p>In the factual scene, the side-road vehicle yields and the ego passes safely. CounterScene identifies that yielding vehicle as the causal variable and compresses its spatiotemporal margin, producing an aggressive cut-in.</p>
</div>

### Scene 905: Rear-End Collision

<div align="center">
<img src="assets/905.png" width="900" />
<p>The trailing vehicle maintains a safe following margin in the factual scene. CounterScene compresses the margin, causing a realistic rear-end crash without distorting the rest of the scene.</p>
</div>

### Scene 913: Head-On Collision

<div align="center">
<img src="assets/913.png" width="900" />
<p>CounterScene applies a minimal spatial intervention that redirects the causally critical vehicle into the ego path, culminating in a realistic head-on collision.</p>
</div>

## Release Status

This repository currently contains the project overview and qualitative assets. The full codebase, trained checkpoints, and evaluation scripts will be released after the paper process is completed.

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{jing2026counterscene,
  title={CounterScene: Counterfactual Causal Reasoning in Generative World Models for Safety-Critical Closed-Loop Evaluation},
  author={Jing, Bowen and Hao, Ruiyang and Zhou, Weitao and Yu, Haibao},
  journal={arXiv preprint arXiv:2603.21104},
  year={2026}
}
```
