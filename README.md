
## Introduction
This is the source code of "Multi-Grained Attention Network with Mutual Exclusion for Composed Query-Based Image Retrieval"

## Multi-Grained Attention Network with Mutual Exclusion for Composed Query-Based Image Retrieval
In this work, we propose a novel Multi-Grained Attention Network with Mutual Exclusion termed MANME for composed query-based image retrieval. The proposed MANME adequately utilize the different granular semantic information via a multi-grained attention network, which encourage the model to capture vital multi-grained visiolinguistic information. And we propose the attention with mutual exclusion module, it adds the mutual exclusion constraint on attention to effectively obtain the proper modified and preserved regions by decreasing the degree of overlap between the preserved and modified regions.
![HVNME](fig/framework_8.30_00.png)
![Constraint](fig/constraint_8.30_00.png)

## Proposed Model (MANME)
* Multi-Grained Attention Network
* Attention with Mutual Exclusion Constraint
* Objective Function
  * Sim-level Matching
  * Attention-level Matching

## The Composed Query-Based Image Retrieval (CQBIR) Task
![Task](fig/task_hebing_00.png)
Illustration of the advantages of composed query-based image retrieval, compared with the general image retrieval (i.e., the top of the figure). The CQBIR task can grants users to express their search intention flexibly. Given a reference image and a modification text, the system retrieves images that resemble the reference image while satisfying the user’s request provided by the text.

## Motivation
![Motivation](fig/intro_sug_00.png)
(a) An illustrative example of the overlapping problem between preserved and modified parts. (b) The illustration of our proposed mutual exclusion constraint. Our goal is to force the modification text to be more relevant to the modified part than to the preserved part, which assists the model to learn the lower degree of overlap.

## Retrieval Examples
![Retrieval](fig/retrieval_8.30_00.png)


## Data Download
The three datasets (FashionIQ, Shoes, Fashion200k) can be downloaded from the official code release for ARTEMIS (ICLR 2022).
## Usage
* Run vocab.py to compute the corresponding vocabulary
  * You should obtain the following vocabulary size for the different datasets:
    - FashionIQ: 3775
    - Shoes: 1330
    - Fashion200K: 4963
* Run train.py (img_finetune, default=False; txt_finetune, default=False)
* Run train.py (img_finetune, default=True; txt_finetune, default=True)


## Results
![Table](fig/res.PNG)

## Acknowledge
We sincerely thank the following works for their provided features and codes.
```bibtex
@inproceedings{delmas2022artemis,
  title={ARTEMIS: Attention-based Retrieval with Text-Explicit Matching and Implicit Similarity},
  author={Delmas, Ginger and Rezende, Rafael S and Csurka, Gabriela and Larlus, Diane},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```