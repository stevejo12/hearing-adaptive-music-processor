
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
Understanding causal vs non-causal approaches is key to understanding the space. Non-causal approaches approaches where the algorithm can "see the future" when predicting how to transform the current timestep. This makes it appropriate for batch processing and non-live processing, but usless for real time situations, like hearing aids and live music. Causal approaches are real time. The algorithm has access to the past and present, but no knowledge of the future. 

Results can be measured in Signal to Distortion ratios (SDR). This is a logarithmic metric measured in decibles. Higher numbers indicate that the correct signal was so many db louder than the distorted noise. Therefore higher is better.

### Motivation
Hearing aid users often have a hard time experiencing music. A survey of hearing aid users (ref 10) found some interesting discrepencies. Most hearing aid users do find that their hearing aids help with hearing music. However, positive feedback was most associated with users with severe or profound hearing loss. These users would probably benefit from any help given by the hearing aid. Furthermore, users with more than 20 years of hearing loss reported benefits compared to users with less than 5 years of hearing loss, who commented on specific negitive experiences listening to music with their hearing aids. The survey hypothesises that those users who report benefits have been deaf for so long, or are so deaf, that they don't remember a pre-hearing aid listening experience. Furthermore, users with some musical training also reported worse experiences.

The results of the survey indicate that while individuals with severe hearing loss don't know they have a problem listening to music, that doesn't mean they won't benefit from  a solution. 

### On the classical music seperation:
The winning paper (ref 11) used an approach called ConvTasNet (ref 12). The results are still pretty crap, which the authors acknowledge. Other methods trained for duets have performed better. Results are better for synthetically created tracks than for real recordings, indicating that something about the synthesised data is not capturing the essence of the real track. Training data is a big issue for classical music, as most recordings take place with all instruments playing at once in a concert hall, where reverberation is present. Seperating specific instruments out of the recording requires training data of that instrument playing on its own. The authors attribute the poor performance to lack of training data. 

Conv-Tas-Net (ref 12) was developed for speech tracking. It is nativly a causal model. It replaces the short time fourior transform using encoder/decoder basis functions to reconstruct the signal of a speaker. Also called a convolutional autoencoder. A PyTorch implementation of this is found in ref 13. 

#### Commentary: 
The authors blame a lack of training data. What if we could encode the existing data in another way which highlights important aspects of melody prediction?

These experiments use standard big training: small validation (train/test) split. However Dr. Neville has us using small training: big testing in the ECE 537 class to build more robust systems. Therefore, an opportunity for our project could be: system with small training set generalizes well to large testing dataset. This would be a challenge which would test our design skills as engineers. 

In the Causal system we can’t look into the future, but we could encode more of the past. The ConvTasNet systems already use memory. Why isn’t it working? What is it about melody recognition that is being miscoded? I’d have to review the training data, but could we get around the instrument misclassification issue by measuring more than the notes (frequencies) at each step? melodies follow other musical elements such as rythm, “feel”, etc which would allow your brain to guess where the next note played by that instrument will be, and adjust if incorrect. So maybe if we encode those elements we could improve seperation. 

Could we do this by just gathering recordings of instrument soloists on spotify and training our model on these? Is it possible that the genre of music would be irrelvant as long as the effects (reverb, etc) on the instrument were the same? Maybe a bit of reverb in the instrument recording is a good thing if the real recordings have reverb. Not unlike assing noise to model training data, a technique which is covered in CSC 503. 

All these systems use very short windows, which would be fair for estimating frequency. However, to track the shape of a melody over time we could consider ways to layer input from longer windows (those long enough to track more than one note...). This could muddy the reconstruction, so they might have to be represented elsewhere in the architecture. This is a suggestion about encoding more memory, which the system already does to some degree. 

The memory system in ref 12 only uses the first and second moments (mean and covariance). Prehaps we should be considering higher order moments, like skew. 

$cLN(\textbf{f}_k) = \frac{\textbf{f}_k - E[\textbf{f}_{t\leq k}]}{\sqrt{Var[\textbf{f}_{t\leq k}] + \epsilon}} \odot \gamma + \Beta$

Could we train a non-causal method to predict causal melodies, in a reinforcement learning type system? It might not be unfoair to do this, after all it's just a pre-trained model...

### ConvTasNet
This is an alternative to using the short-time fast fourier transform. 
---

## Timeline

| Milestone | Target Date | Owner |
|-----------|-------------|-------|
| Design specification complete | 13 Oct 26 | All|
| Dataset loaded and preprocessing pipeline working | 28 Oct 26 | Steve | 
| Setup Cadenza Baseline Code | 28 Oct 26 | Agam |
| Review open-source projects and bring them into our GitHub repo| 28 Oct 26 | Likkhith |
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

## Feedback on task breakdown
The project design specification is clearly written. The task separation is reasonable and using the Cadenze challenge as the main focus is good. The objectives and performance indicators are ok but the advanced ones are more basic/expected in terms of difficulty. Although there is a lot of setup such as reading files/metadata etc I expect at least for the advanced performance indicators a bit more algorithmic stuff. Some suggestions could be **implementing different published algorithms about the tasks**. Also **one member could focus on pyClarity and hearing loss simulation in general**. *I encourage you to reach out to Kirill Abrosimov and he can provide you more guidance.*

### Ideas on feedback / questions we can ask Krill:
 - Should we focus on both tasks in one month? (ie Lyric Intelligibility and Classical Music rebalancing)
 - Maybe instead of original algorithm we just implement existing algorithms and do a comparitive analysis
## Team and Roles

### Agam Kamran
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
### Steve Vinsensius Jo 
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
### Mackenzie Mallett
### Objective 1: Literature Review
- **PI1 (basic):** Review the winning papers from the 2024/2025 CAD2 challenge.
- **PI2 (basic):** Review keystone literature in music retreival algorithm design.
- **PI3 (expected):** Summarize common approaches and techniques from the literature in a one-pager.
- **PI4 (expected):** Construct an algorithm design template with key issues.
- **PI5 (advanced):** Draft algorithm roadmap based on previous literature.

### Objective 2: Implement the ConvTasNet Method used by the winning paper
- **PI1 (basic):** Download and implement PyTorch (this is what the original setup was done in) or TensorFlow.
- **PI2 (expected):** Setup the Conv-TasNet architecture using PyTorch or Tensor Flow.
- **PI3 (expected):** Optimize hyperparameters to mimic the performance of the winning paper.
- **PI4 (basic):** Test on the provided dataset to determine baseline.
- **PI5 (advanced):** Analyze which blocks of the Convolutional Network have the biggest impact on performance.

### Objective 3: Adjust the Conv-TasNet method to incorporate elements of music not captured by the current setup
- **PI1 (basic):** Identify which elements of the network would encode non-frequency elements (encoder and spereation stages?)(maybe a longer time window to improve accuracy as the melody progresses.).
- **PI2 (advanced):** Consider how to implement factors such as: Rhythm/Tempo, Articulation/Duration, Space/Rest, Phrasing, etc into the network. The idea is to identify each signal by more than just the frequency range (the direction of the melody).
- **PI3 (basic):** Implement elements of the network to encode non-frequency elements.
- **PI4 (expected):** Test selected methods to determine which have the greatest impact (it may be just one...).
- **PI5 (expected):** Per-instrument experiments.

---
### Likkhith Krishnamurthy Aditya
### Objective 1: Review open-source projects, add them to our repo, and pair-program with Agam
- **PI1 (basic):** Review open-source separation projects (method, licence, fit for Task 2).
- **PI2 (basic):** Add the useful ones to `third_party/` with licences and a summary table.
- **PI3 (expected):** Pair-program with Agam to set up the CAD2 data and Clarity in Colab.
- **PI4 (expected):** Pair-program with Agam to run the baseline and HAAQI on a validation scene.
- **PI5 (advanced):** Compare the causal and non-causal separators on 20 scenes with HAAQI and SDR.

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
3. Development of an adaptive test of musical scene analysis abilities for normal-hearing and hearing-impaired listeners, Hake et al, 2023. 
4. Music listening and hearing aids: perspectives from audiologists and their patients, Greasley et al, 2020
5. Perception of Recorded Music With Hearing Aids - Compression Differentially Affects Musical Scene Analysis and Musical Sound Quality - Hake et al 2025
6. Can You Hear Out the Melody? Testing Musical Scene Perception in Young Normal-Hearing and Older Hearing-Impaired Listeners, Siedenburg et al, 2020
7. Remixing Preferences for Western Instrumental Classical Music of Bilateral Cochlear Implant Users, Althoff et al, 2024
8. Remixing music using source separation algorithms to improve the musical experience of cochlear implant users, Pons et al 2016
9. Exploring level- and spectrum-based music mixing transforms for hearing-impaired listeners,  Benjamin & Siedenburg, 2023
10. Music and Hearing Aids - Madsen and Moore 2014

11. Source Separation of Small Classical Ensembles Challenges and Opportunities - Roa-Dabike et al 2025 (Winning classial music seperation paper)
12. Conv-TasNet Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation -  Luo and Mesgarani (2019)
13. https://github.com/naplab/Conv-TasNet

<!-- target 15–20 -->
## Lit Review Scratch
### Lyric Intelligibility
Lyrics are a key part of musical appreciation (Greasley et al., 2013)
Hearing out lyrics common problem for those with HL (Greasley et al., 2020)
