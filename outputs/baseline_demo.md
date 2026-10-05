# Baseline Demo: Zero-Shot BioGPT, Greedy vs. Sampling

Base model: `microsoft/biogpt` (no fine-tuning). Prompts are synthetic/templated -- no real patient data.

**Point being demonstrated:** greedy decoding (`do_sample=False`) is deterministic -- rerunning it produces the exact same text every time. Sampling (`do_sample=True, top_k=50, top_p=0.95`) is stochastic -- three runs with three different seeds below produce three different continuations from the identical prompt. For a system generating clinical text, that variance is a real design consideration: reproducibility vs. diversity of phrasing.

## Prompt 1
> Patient presented with shortness of breath and elevated blood pressure. Diagnosis: 

**Greedy (do_sample=False), run twice to confirm determinism:**

```
Patient presented with shortness of breath and elevated blood pressure. Diagnosis:  a case report.
```
*Identical on rerun: True*

**Sampling (do_sample=True, top_k=50, top_p=0.95), 3 runs with different seeds:**

*seed=0:*
```
Patient presented with shortness of breath and elevated blood pressure. Diagnosis:  giant cell arteritis.
```
*seed=1:*
```
Patient presented with shortness of breath and elevated blood pressure. Diagnosis:  pheochromocytoma.
```
*seed=2:*
```
Patient presented with shortness of breath and elevated blood pressure. Diagnosis:  Primary Adrenal Insufficiency.
```
*All 3 sampling runs identical: False (expected: False)*

## Prompt 2
> A 65-year-old male admitted with chest pain radiating to the left arm. Hospital course: 

**Greedy (do_sample=False), run twice to confirm determinism:**

```
A 65-year-old male admitted with chest pain radiating to the left arm. Hospital course:  acute myocardial infarction (AMI).
```
*Identical on rerun: True*

**Sampling (do_sample=True, top_k=50, top_p=0.95), 3 runs with different seeds:**

*seed=0:*
```
A 65-year-old male admitted with chest pain radiating to the left arm. Hospital course:  pain relief was brief.
```
*seed=1:*
```
A 65-year-old male admitted with chest pain radiating to the left arm. Hospital course:  ECG showed ST segment elevation in leads II, III and aVF, and cardiac enzyme levels were elevated, although the levels of troponin-I and creatine kinase-MB were normal.
```
*seed=2:*
```
A 65-year-old male admitted with chest pain radiating to the left arm. Hospital course:  he was normotensive and nonsmoker.
```
*All 3 sampling runs identical: False (expected: False)*

## Prompt 3
> Patient with type 2 diabetes admitted for diabetic ketoacidosis. Treatment included 

**Greedy (do_sample=False), run twice to confirm determinism:**

```
Patient with type 2 diabetes admitted for diabetic ketoacidosis. Treatment included  insulin therapy, fluid therapy, and bicarbonate infusion.
```
*Identical on rerun: True*

**Sampling (do_sample=True, top_k=50, top_p=0.95), 3 runs with different seeds:**

*seed=0:*
```
Patient with type 2 diabetes admitted for diabetic ketoacidosis. Treatment included  strict metabolic control, intensive insulin therapy (target blood glucose levels 80-110 mg / dl), and fluids were given to maintain urine output.
```
*seed=1:*
```
Patient with type 2 diabetes admitted for diabetic ketoacidosis. Treatment included  intravenous fluids and bicarbonate.
```
*seed=2:*
```
Patient with type 2 diabetes admitted for diabetic ketoacidosis. Treatment included  intravenous fluids and insulin.
```
*All 3 sampling runs identical: False (expected: False)*
