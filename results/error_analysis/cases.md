# Error analysis: final model on the test split

## 1. Missing information treated as a negative

- Annotation: 405 (exclusion)
- Criterion: Major surgery within 3 weeks prior to study entry.
- Expert: **not enough information**, citing none
- Model: **not excluded**, citing none
- Source of the problem: Model + labelling convention
- Why: The note never mentions surgery. The expert says this can't be decided; the model takes the silence to mean the patient had none.

## 2. Labelling convention for unmentioned rules

- Annotation: 425 (exclusion)
- Criterion: APACHE II (health) score > 25
- Expert: **not excluded**, citing none
- Model: **not enough information**, citing none
- Source of the problem: Dataset
- Why: An APACHE II score can't be worked out from this note, and the note doesn't state one, so the model's caution is reasonable. The expert marks an unmentioned exclusion as not excluded here, but as not enough information elsewhere.

## 3. Over-cautious despite direct evidence

- Annotation: 766 (inclusion)
- Criterion: Fever
- Expert: **included**, citing [0] "A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes."
- Model: **not enough information**, citing [0] "A 5-year-old boy presents to the emergency department with complaints of progressively worsening dysphagia, drooling, fever and vocal changes."
- Source of the problem: Model
- Why: The note says fever outright and the model cites the right sentence, yet still declines to decide.

## 4. Medical term needs interpreting

- Annotation: 186 (inclusion)
- Criterion: Heart rate of at least 90/min;
- Expert: **included**, citing [2] "On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic."
- Model: **not enough information**, citing [3] "Her body temperature is normal and her blood pressure is 80/60."
- Source of the problem: Model
- Why: Tachycardic means a heart rate above 100, so at least 90 is met. The model did not make that link and cited the blood-pressure sentence.

## 5. Numeric threshold needs reasoning

- Annotation: 188 (inclusion)
- Criterion: The use of vasopressors to maintain a systolic blood pressure of at least 90mmHg despite fluid resuscitation, low dose of dopamine (≤ 5 μg/kg per minute), or dobutamine;
- Expert: **not included**, citing [3] "Her body temperature is normal and her blood pressure is 80/60."
- Model: **not enough information**, citing [3] "Her body temperature is normal and her blood pressure is 80/60."
- Source of the problem: Model
- Why: A blood pressure of 80/60 shows systolic pressure is not being held at 90 or above. The model cites the right sentence without drawing the conclusion.

## 6. Not applicable vs not enough information

- Annotation: 834 (inclusion)
- Criterion: --Negative pregnancy test for females
- Expert: **not applicable**, citing none
- Model: **not enough information**, citing none
- Source of the problem: Model
- Why: A pregnancy test can't apply to a 4-year-old boy. The model treats this as missing data, not as a rule that doesn't apply.

## 7. Applicability, labelled inconsistently

- Annotation: 322 (exclusion)
- Criterion: Pregnant women or women of childbearing age who do not have adequate contraception.
- Expert: **not excluded**, citing [0] "A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40."
- Model: **not applicable**, citing none
- Source of the problem: Dataset + model
- Why: The patient is male. The expert says not excluded, citing his sex; the model says not applicable. Case 834 shows experts also use not applicable for this kind of rule.

## 8. Negated exclusion rule

- Annotation: 414 (exclusion)
- Criterion: Patients without systemic Lupus erythematosus
- Expert: **not excluded**, citing [0] "A 21-year-old female is evaluated for progressive arthralgias and malaise."; [1] "On examination she is found to have alopecia, a rash mainly distributed on the bridge of her nose and her cheeks, a delicate non-palpable purpura on her calves, and swelling and tenderness of her wrists and ankles."; [2] "Her lab shows normocytic anemia, thrombocytopenia, a 4/4 positive ANA and anti-dsDNA."; [3] "Her urine is positive for protein and RBC casts."
- Model: **not enough information**, citing [1] "On examination she is found to have alopecia, a rash mainly distributed on the bridge of her nose and her cheeks, a delicate non-palpable purpura on her calves, and swelling and tenderness of her wrists and ankles."; [2] "Her lab shows normocytic anemia, thrombocytopenia, a 4/4 positive ANA and anti-dsDNA."
- Source of the problem: Model
- Why: The rule excludes patients without lupus, and the note strongly suggests lupus, so the patient is not excluded. The double negative defeats the model, which cites lupus findings but won't decide.

## 9. Contradictory information in the note

- Annotation: 827 (inclusion)
- Criterion: Clinical diagnosis of bronchial asthma
- Expert: **not enough information**, citing [0] "A 4 year old boy presents to the emergency room with wheezing."; [1] "He has had a history of allergic rhinitis, but no history of wheezing."; [2] "His mother reports that 5 hours ago patient was playing in the backyard sandbox when she heard him suddenly start coughing."; [3] "The coughing lasted only moments, but he has been audibly wheezing since."; [6] "Wheezing is heard in the mid-right chest area."
- Model: **not included**, citing [0] "A 4 year old boy presents to the emergency room with wheezing."; [1] "He has had a history of allergic rhinitis, but no history of wheezing."
- Source of the problem: Model
- Why: The boy is wheezing now but has no history of wheezing, and it began suddenly while playing, so asthma is neither confirmed nor ruled out. The model picks one side, citing the no-history sentence.

## 10. Unsupported evidence

- Annotation: 410 (exclusion)
- Criterion: Other significant, unstable medical disease not related to SLE that in the investigator's opinion would confound the study result or put the patient at risk.
- Expert: **not enough information**, citing none
- Model: **not excluded**, citing [0] "A 21-year-old female is evaluated for progressive arthralgias and malaise."; [1] "On examination she is found to have alopecia, a rash mainly distributed on the bridge of her nose and her cheeks, a delicate non-palpable purpura on her calves, and swelling and tenderness of her wrists and ankles."; [2] "Her lab shows normocytic anemia, thrombocytopenia, a 4/4 positive ANA and anti-dsDNA."
- Source of the problem: Model
- Why: The cited sentences describe lupus symptoms, while the rule asks about other diseases not related to lupus. The citations don't support the answer.

## 11. Only part of a compound rule is met

- Annotation: 323 (inclusion)
- Criterion: Immune Deficiency and early BMF in childhood
- Expert: **not enough information**, citing [0] "A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40."
- Model: **included**, citing [0] "A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40."
- Source of the problem: Model
- Why: The rule needs immune deficiency and bone marrow failure in childhood. The note shows only immune deficiency, in a 48-year-old, but the model accepts it.

## 12. Right label, incomplete evidence

- Annotation: 413 (inclusion)
- Criterion: Patients with systemic Lupus erythematosus
- Expert: **included**, citing [0] "A 21-year-old female is evaluated for progressive arthralgias and malaise."; [1] "On examination she is found to have alopecia, a rash mainly distributed on the bridge of her nose and her cheeks, a delicate non-palpable purpura on her calves, and swelling and tenderness of her wrists and ankles."; [2] "Her lab shows normocytic anemia, thrombocytopenia, a 4/4 positive ANA and anti-dsDNA."; [3] "Her urine is positive for protein and RBC casts."
- Model: **included**, citing [0] "A 21-year-old female is evaluated for progressive arthralgias and malaise."; [1] "On examination she is found to have alopecia, a rash mainly distributed on the bridge of her nose and her cheeks, a delicate non-palpable purpura on her calves, and swelling and tenderness of her wrists and ankles."; [2] "Her lab shows normocytic anemia, thrombocytopenia, a 4/4 positive ANA and anti-dsDNA."
- Source of the problem: Model
- Why: The label is correct, but the model misses sentence 3, the kidney finding, which the expert also counts.

## 13. Right label, cites beyond the expert

- Annotation: 384 (inclusion)
- Criterion: Laboratory values as follows Hemoglobin ≥ 100 g/L Absolute neutrophil count ≥ 1.0 x 109/L Total bilirubin ≤ 1.5 x upper limit of normal (ULN) AST (SGOT) / ALT (SGPT) ≤ 2.5 x ULN
- Expert: **not enough information**, citing none
- Model: **not enough information**, citing [2] "Her lab shows normocytic anemia, thrombocytopenia, a 4/4 positive ANA and anti-dsDNA."
- Source of the problem: Labelling convention
- Why: Not enough information is correct. The model cites the lab sentence, which is relevant but doesn't give the needed values; the expert cites nothing.
