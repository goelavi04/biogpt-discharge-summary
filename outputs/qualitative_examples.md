# Qualitative Examples

3 highest-ROUGE-L and 3 lowest-ROUGE-L fine-tuned (greedy) generations on the test set, by `ft_greedy` vs. `reference` ROUGE-L. Hallucination flags use the same approximate lexical proxy described in outputs/metrics.json -- NOT a validated clinical fact-checker.

# Good examples
## Good (id=mts_3541, ROUGE-L=0.238)

**Reference:**
```
The patient was admitted to the hospital for further evaluation. A diagnostic procedure was performed.
```
**Fine-tuned generation (greedy):**
```
The patient was admitted to the hospital for further evaluation. The patient was started on IV fluids and IV antibiotics. The patient's pain was relieved and his bowel movements were improved. The patient also had a nasogastric tube placed. The patient had a nasogastric tip placed and the nasogastric tube was removed. The patient did not have any episodes of vomiting or diarrhea. The patient continued to have abdominal pain and was started on oral antibiotics. The pain resolved and the patient was discharged home.
```

## Good (id=mts_2581, ROUGE-L=0.222)

**Reference:**
```
The patient was admitted on 09/04/2007 to undergo total vaginal hysterectomy. The procedure preceded as planned without complication. Uterus was sent for pathologic analysis. The patient was monitored in the hospital, 2 days postoperatively. She recovered quite well and vitals remained stable.
Laboratory studies, H&H were followed and appeared stable on 09/05/2007 with hemoglobin of 11.2 and hematocrit of 31.8.
The patient was ready for discharge on Monday morning of 09/06/2007.
```
**Fine-tuned generation (greedy):**
```
The patient was admitted to the hospital for further evaluation. The patient had an uncomplicated postoperative course. The patient's postoperative course was unremarkable. The patient returned to the office on the second postoperative day. The patient did not complain of any pain or discomfort. The patient is being followed up in the outpatient clinic.
```

## Good (id=mts_2614, ROUGE-L=0.220)

**Reference:**
```
The patient was admitted to the surgical suite and taken to the operating room where a dilatation and curettage (D&C) was performed. Hysteroscopy revealed a large endometrial cavity fibroid. Careful shaving and excision of this fibroid was performed with removal of the fibroid. Hemostasis was noted completely at the end of this procedure. Postoperatively, the patient has done well. The patient was given a 2nd unit of packed red blood cells because of intraoperative blood loss. The patient is now ambulating without difficulty and tolerating her diet. The patient desires to go home. The patient is discharged to home.
```
**Fine-tuned generation (greedy):**
```
The patient was admitted to the hospital for evaluation of menometrorhagia unresponsive for medical therapy. She was admitted to Dr. X p.R.n. and Dr. X-Ray. She was given a diagnostic hysteroscopy. The procedure was successful. The patient's hematocrit improved to 25, hemoglobin was 8.3. The patient returned to normal activity. The patient had no complications. The patient did not require any further transfusions. The patient tolerated the procedure well. She was discharged home. She was able to return to work. She was advised to return to her family. She was also advised to return for a follow-up visit. She was sent to Dr. X-ray. The patient showed no evidence of a uterine fibroid. The patient said that she would have a repeat hysteroscopy if she had a similar procedure.
```

# Bad examples
## Bad (id=mts_3387, ROUGE-L=0.130)

**Reference:**
```
The child was observed on IV fluids and advanced to clear liquids and then regular diet as tolerated. On the second hospital day, mother was comfortable taking her to home. Mother did not have a good explanation for the weight loss. In the hospital, her weight was 124 pounds, her height 58 inches, temperature 98.0 degree F., pulse 123, respirations 18, blood pressure 148/94. Follow up blood pressure were some of them were in the 125 to 70 range making us think her hypertension as labile and perhaps related to the excitement of the admission.
She seem quite happy and in no distress at the time of discharge. We will follow up in the office and try to further evaluate her for the unexplained weight loss. She has been taking the Adderall for at least a year, and the mother does not think the Adderall is the cause of the weight loss. The free T4 is borderline high and probably bears repeating along with further studies for Graves disease as an outpatient.
```
**Fine-tuned generation (greedy):**
```
The patient was admitted to the hospital and given MiraLax.She was started on oral clonidine and Adderall. She was started to take Adderall for her hyperactivity and constipation. She was also started on laxatives. She was given a bowel preparation and was started on a diet. She was then started on a low-residue diet. She had a good response to this. She was discharged home on a diet and laxative.
```
*No proxy-flagged terms.*

## Bad (id=mts_3852, ROUGE-L=0.120)

**Reference:**
```
The patient is a 44-year-old African-American male who was diagnosed with end-stage renal disease requiring hemodialysis three times per week approximately four to five months ago. He reports that over the past month, he has been feeling lightheaded when standing and has had three syncopal episodes during this time with return of consciousness after several minutes. He reportedly had this even while seated and denied overt dizziness. He reports this lightheadedness is made even worse when standing. He has had these symptoms almost daily over the past month. He does report some confusion when he awakens. He reports that he loses consciousness for two to three minutes. Denies any bowel or bladder loss, although he reports very little urine output secondary to his end-stage renal disease. He denied any palpitations, warmth, or diaphoresis, which is indicative of vasovagal syncope. There were no witnesses to his syncopal episodes. He also denied any clonic activity and no history of seizures. In the emergency room, the patient was given fluids and orthostatics were checked. At that time, orthostatics were negative; however, due to the fact that fluid had been given before, it is impossible to rule out orthostatic hypotension. The patient presented to the hospital on Coreg 12.5 mg b.i.d. and lisinopril 10 mg daily secondary to his hypertension, congestive heart failure with dilated cardiomyopathy and end-stage renal disease. Regarding his syncopal episodes, he was admitted with likely orthostatic hypotension. Cardiology was consulted and their recommendations were to reduce the lisinopril to 5 mg daily. At that time, the Coreg had been held secondary to hypotension. Cardiology also ordered a nuclear medicine myocardial perfusion stress test. Regarding the end-stage renal disease, Nephrology was consulted as the patient was due for hemodialysis treatment the day following admission. Nephrology was able to perform dialysis on the patient and Renal concurred that the presyncopal symptoms were likely due to decreased intravascular volume in the postdialytic time frame. Renal agreed with decreasing his lisinopril to 5 mg daily and decreasing the Coreg to 6.25 mg b.i.d. They reported that the Procrit should be continued. As previously indicated regarding the dilated cardiomyopathy, Cardiology ordered a nuclear medicine stress test to be performed. Also, regarding the patient's hypertension, he actually was noted to have hypotension on admission, and as previously stated, the Coreg was originally discontinued and then it was restarted at 6.25 mg b.i.d. and the patient tolerated this well. The patient's hospital course remained uncomplicated until September 17, 2007, the day the nuclear medicine stress test was scheduled. The patient stated that he was reluctant to proceed with the test and he was afraid of needles and the risks associated with the test although the procedure was explained to the patient and the risks of the procedure were quit low, the patient proceeded to discharge himself against medical advice.
```
**Fine-tuned generation (greedy):**
```
The patient was admitted to the hospital for evaluation of syncope. He was admitted to a telemetry unit. He was tachycardic and hypotensive. He was treated with IV fluids and IV nitroglycerin. He was started on IV fluids and nitroglycerin. He had a good response. He was transferred to the intensive care unit. He had an episode of syncope. His labs were unremarkable. He was placed on IV fluids, IV nitroglycerin, and IV fluids and oral nitroglycerin. His labs remained unremarkable. He had no episodes of syncope. The patient was transferred to a cardiology service.
```
*No proxy-flagged terms.*

## Bad (id=mts_1899, ROUGE-L=0.090)

**Reference:**
```
At the time of transfer to ABCD, these were the following issues.
```
**Fine-tuned generation (greedy):**
```
The baby was born at the ABCD Hospital on 07 / 12 / 2006. The mother was a 20-years-old gravidus 1, para 1, who received prenatal treatment for preeclampsia and was admitted to the hospital for evaluation of fetal distress. The mother had a history of hypertension and diabetes mellitus. The mother's condition was complicated by preeclampsia and was treated with magnesium sulfate. The baby was delivered at the ABCDC Hospital on 07 (12 / 15 / 2006).
```
**Terms in generation not found in source note (proxy flag, verify manually):**
`ABCDC`
