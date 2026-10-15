
# SP30 - Hearing-Adaptive Music Processor
> CSC 575 | University of Victoria | Fall 2026

---

## Overview


The [2nd Cadenza challenge - Task 2: Rebalancing Classical Music](https://cadenzachallenge.org/docs/cadenza2/Rebalancing%20Classical/rebalancing) improves music for people with hearing loss. From the challenge description: "Hearing loss causes various problems such as quieter music passages being inaudible, poor and anomalous pitch perception, difficulties identifying and picking out instruments, and problems hearing out lyrics." Task 2, which we will focus on for our project, focuses on rebalancing the levels of insturnments within a classical music sample to allow personal remixing by hearing aid users. 

The goal is to build a tool that takes a music file and the listener's hearing loss profile, then reshapes the audio to compensate -  so someone with partial hearing experiences music as close to its intended sound as possible.

---

## Motivation

People with hearing loss don't just hear music more quietly - they hear it differently. Certain frequencies drop out, distorting the balance of instruments, vocals, and harmonics that the artist intended. 

---

## Proposed Approach

To build our tool we will rely heavily on the datasets and evaluation metrics available in the [Cadenza CAD2 Challenge: Task 2](https://cadenzachallenge.org/docs/cadenza2/intro). This challenge focuses on rebalanceing elements of classical music samples for hearing aid users.

The Task 2 Dataset (https://zenodo.org/records/14335552) contains music samples from a woodwind ensemble in the BACH10 and URMP datasets, and listner metadata for 50 real hearing aid users. Both are split into training, validation, and testing datasets. 

The evaluation of our algorithm will be conducted as follows:
1. A scene generator selects the music signal and target gains for each insturnment in the ensemble.
2. The music enhancement stage (our algorithm) takes the music as input and attempts the desired remix.
3. Our algorithm also takes as input listener audiograms and compressor settings to allow personalised processing.
4. The output of the algorithmm is evaluated for audio quality using the Hearing-Aid Audio Quality Index.

In desiging our algorithm we have the benefit of knowing the previous results of this challenge. We can use the top papers to inform our approach and highlight areas for investigation. A thorough liturature review is therefore the first step of this project. 

After reviewing previous work we will put forward a draft design specification for our algorithm. The constraints for this algorithm will be the same as those used by the CAD2 challenge, and evaluation will be the same as that which is described above.

If time permits, we will attempt to package our algorithm into a practical tool available to avid music listeners experiencing hearing loss. 

## Tools and Libraries

| Tool | Purpose |
|------|---------|
| clarity (Cadenza toolkit) |MSBG hearing loss simulator, NAL-R fitting, HAAQI metric|
| librosa | Spectrogram visualization + Feature Extraction|
| soundfile | Reading and writing WAV files from the CAD2 dataset |
| NumPy | Array Math |

More to be expanded upon as we approach this project.

---

## Datasets

| Dataset | Source | Usage |
|---------|--------|-------|
| Cadenza Challenge Datasets | [Zenodo](https://zenodo.org/records/14335552) | |

---

## Related Work

This section will be expanded in parrelel with the liturature review described in "Proposed Approach".

---

## Timeline

| Milestone | Target Date | Owner |
|-----------|-------------|-------|
| Design specification complete | 13 Oct 26 | All|
| Dataset loaded and preprocessing pipeline working | 28 Oct 26 | Steve | 
| Setup Cadenza Baseline Code | 28 Oct 26 | Agam |
| Build Evaluation Metrics as a Module | 28 Oct 26 | Likkhith |
| Literature Review| 28 Oct 26 | Mack |
| Progress report | 31 Oct 26 | All |
| Algorithm Design | 20 Nov 26|All |
|&nbsp;&nbsp;↳  Algo Milestone 1 | 6 Nov 26 | All |
|&nbsp;&nbsp;↳&nbsp;&nbsp;↳  Task 1.1 - TBD | 6 Nov 26 | Agam |
|&nbsp;&nbsp;↳&nbsp;&nbsp;↳  Task 1.2 - TBD | 6 Nov 26 | Likkhith |
|&nbsp;&nbsp;↳&nbsp;&nbsp;↳  Task 1.3 - TBD | 6 Nov 26 | Mack |
|&nbsp;&nbsp;↳&nbsp;&nbsp;↳  Task 1.4 - TBD | 6 Nov 26 | Steve |
|&nbsp;&nbsp;↳  Algo Milestone 2 | 13 Nov 26 | All |
|&nbsp;&nbsp;↳&nbsp;&nbsp;↳  Task 2.1 - TBD | 13 Nov 26 | Agam |
|&nbsp;&nbsp;↳&nbsp;&nbsp;↳  Task 2.2 - TBD | 13 Nov 26 | Likkhith |
|&nbsp;&nbsp;↳&nbsp;&nbsp;↳  Task 2.3 - TBD | 13 Nov 26 | Mack |
|&nbsp;&nbsp;↳&nbsp;&nbsp;↳  Task 2.4 - TBD | 13 Nov 26 | Steve |
|&nbsp;&nbsp;↳  Algo Milestone 3 | 20 Nov 26 | All |
|&nbsp;&nbsp;↳&nbsp;&nbsp;↳  Task 3.1 - TBD | 20 Nov 26 | Agam |
|&nbsp;&nbsp;↳&nbsp;&nbsp;↳  Task 3.2 - TBD | 20 Nov 26 | Likkhith |
|&nbsp;&nbsp;↳&nbsp;&nbsp;↳  Task 3.3 - TBD | 20 Nov 26 | Mack |
|&nbsp;&nbsp;↳&nbsp;&nbsp;↳  Task 3.4 - TBD | 20 Nov 26 | Steve |
| Scene Generation and Random Listener Selection | | | 
| Spectrogram visualization (before/after) | 30 Nov 26| |
| Final report  | 30 Nov 26| All |

---

## Team and Roles

### Agam Kamran - V00946178
### Objective 1 (by Oct 28): Setup Cadenza Baseline Code
- **PI1 (basic):** Clone the Clarity repository and confirm the CAD2 Task 2 environment runs without errors in a Google Colab notebook.
- **PI2 (basic):** Load a single scene from the CAD2 dataset
- **PI3 (expected):** Run the full baseline pipeline end-to-end on one scene
- **PI4 (expected):** Compute the HAAQI score for the baseline output on that scene
- **PI5 (advanced):**: Profile the baseline pipeline (runtime, memory per stage)

### Objective 2: 
- **PI1 (basic):**
- **PI2 (basic):**
- **PI3 (expected):**
- **PI4 (expected):**
- **PI5 (advanced):**

### Objective 3: 
- **PI1 (basic):**
- **PI2 (basic):**
- **PI3 (expected):**
- **PI4 (expected):**
- **PI5 (advanced):**

---
### Steve Vinsensius Jo V01099466
### Objective 1: Dataset loaded and preprocessing pipeline working
- **PI1 (basic):** Download the CAD2 Task 2 training and validation data from Zenodo and verify against md5 checksums.
- **PI2 (basic):** Load the listener, scene, gain, and compressor JSON files into Python.
- **PI3 (expected):** Convert the audio to a consistent format and cut it into the fixed-length clips used by the scenes.
- **PI4 (expected):** Write a documented data loader that returns the mixture, stems, audiogram, and gains for a given scene.
- **PI5 (advanced):** Load the real-recording tuning set and report summary statistics comparing it with the synthetic data.

### Objective 2: 
- **PI1 (basic):**
- **PI2 (basic):**
- **PI3 (expected):**
- **PI4 (expected):**
- **PI5 (advanced):**

### Objective 3:
- **PI1 (basic):**
- **PI2 (basic):**
- **PI3 (expected):**
- **PI4 (expected):**
- **PI5 (advanced):**

---
### Mackenzie Mallett - V00878418	
### Objective 1: Literature Review
- **PI1 (basic):** Review the winning papers from the 2024/2025 CAD2 challenge.
- **PI2 (basic):** Review keystone literature in music retreival algorithm design.
- **PI3 (expected):** Summarize common approaches and techniques from the literature in a one-pager.
- **PI4 (expected):** Construct an algorithm design template with key issues.
- **PI5 (advanced):** Draft algorithm roadmap based on previous literature.

### Objective 2:
- **PI1 (basic):**
- **PI2 (basic):**
- **PI3 (expected):**
- **PI4 (expected):**
- **PI5 (advanced):**

### Objective 3:
- **PI1 (basic):**
- **PI2 (basic):**
- **PI3 (expected):**
- **PI4 (expected):**
- **PI5 (advanced):**

---
### Likkhith Krishnamurthy Aditya - V01098937
### Objective 1: Build Evaluation Metrics as a Module
- **PI1 (basic):** Install pyClarity 0.6.3 and compute HAAQI for one output signal.
- **PI2 (basic):** Wrap HAAQI in a reusable function that returns left, right and average scores.
- **PI3 (expected):** Add per-instrument SDR using museval.
- **PI4 (expected):** Score a batch of scenes and save the results to a CSV file.
- **PI5 (advanced):** Confirm the scores match the official CAD2 evaluation script on a few scenes.

### Objective 2:
- **PI1 (basic):**
- **PI2 (basic):**
- **PI3 (expected):**
- **PI4 (expected):**
- **PI5 (advanced):**

### Objective 3:
- **PI1 (basic):**
- **PI2 (basic):**
- **PI3 (expected):**
- **PI4 (expected):**
- **PI5 (advanced):**

---

## Bibliography

1. https://cadenzachallenge.org/docs/cadenza2/intro
2. https://github.com/claritychallenge/clarity
3. TBD
4. TBD
5. TBD
.
.

20
<!-- target 15–20 -->
