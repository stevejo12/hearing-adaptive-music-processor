# Third-party projects

Open-source projects reviewed for CAD2 Task 2 (rebalancing classical ensembles).
Licences were checked in each repository on 11 November 2026.

## Summary

| Project | Method | Licence | Training code | Fit for Task 2 | How we use it |
|---|---|---|---|---|---|
| [pyClarity](https://github.com/claritychallenge/clarity) | CAD2 baseline: 8 per-instrument ConvTasNet separators (causal and non-causal), HAAQI, compressor | MIT | Yes (`recipes/cad2/task2/ConvTasNet`) | Ready: trained on classical instruments | Cloned by our Colab notebooks |
| [naplab/Conv-TasNet](https://github.com/naplab/Conv-TasNet) | Original Conv-TasNet model definition (speech) | CC BY-NC-SA 3.0 US (in its README) | No | Reference only: no training code or weights | Copied to `third_party/Conv-TasNet` |
| [kaituoxu/Conv-TasNet](https://github.com/kaituoxu/Conv-TasNet) | PyTorch reproduction of Conv-TasNet | MIT | Yes | Needs training on CAD2 stems | Candidate to copy |
| [Open-Unmix](https://github.com/sigsep/open-unmix-pytorch) | Spectrogram LSTM separator, accepts custom datasets | MIT | Yes | Needs training on CAD2 stems | `pip install openunmix` (PI5) |
| [Demucs (HTDemucs)](https://github.com/facebookresearch/demucs) | Hybrid transformer separator, pretrained on pop | MIT | Yes | Pop stems only (vocals, drums, bass, other); no longer maintained | `pip install demucs` (PI3) |
| [KUIELab-MDX-Net](https://github.com/kuielab/mdx-net) | Spectrogram U-Net; part of the ICASSP 2024 Cadenza winner's ensemble | MIT | Yes | Needs training | Candidate to copy |
| [DTTNet](https://github.com/junyuchen-cjy/DTTNet-Pytorch) | Dual-path TFC-TDF UNet; part of the ICASSP 2024 Cadenza winner's ensemble | Apache 2.0 | Yes | Needs training | Candidate to copy |
| [Asteroid](https://github.com/asteroid-team/asteroid) | Toolkit: ConvTasNet, DPRNN and training recipes | MIT | Yes | Needs training | `pip install asteroid` if needed |
| [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training) | One framework for BS-RoFormer, SCNet, MDX23C and others | MIT | Yes | Needs training; GPU required | Link only (large) |
| [BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) | Band-split RoFormer model | MIT | Model only | Needs training | Link only |
| [museval](https://github.com/sigsep/sigsep-mus-eval) | BSS Eval v4 (SDR) | MIT | n/a | Scoring separated instruments | `pip install museval` (PI4) |
| [JusperLee/Conv-TasNet](https://github.com/JusperLee/Conv-TasNet) | Conv-TasNet in PyTorch and Lightning | None stated | Yes | Needs training | Link only (no licence, do not copy) |
| [HAAQI-Net](https://github.com/dyahayumgw/HAAQI-Net) | Neural estimate of HAAQI | None stated | n/a | Faster scoring | Link only (no licence, do not copy) |

## Findings

- Only the pyClarity CAD2 models separate classical instruments out of the box. Every other separator above was built for pop stems or speech and needs training on CAD2 stems.
- The CAD2 results page links no code for the Task 2 entries, and the ICASSP 2024 winner (an average of HDemucs, MDX-Net and DTTNet fine-tuned on challenge data) released no code, so their methods have to be reimplemented.

## Rules for this folder

- Copy code here only if its licence allows it, and keep its LICENSE or licence notice.
- Add a `SOURCE.md` to each copied project with the original URL, commit and authors.
- Projects with no licence stay as links only.
