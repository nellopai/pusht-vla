# Reading list

In reading order. Tick a box when you've read it, and add your own notes underneath each item.

## 1. Imitation learning basics (~1 h)

- [ ] **Berkeley CS285, Lecture 2: Imitation Learning** (Sergey Levine)
  https://rail.eecs.berkeley.edu/deeprlcourse/
  Watch the video or read the slides. The core idea: learn a policy by copying
  demonstrations (behaviour cloning). The core problem: small errors compound over
  time, so the agent drifts into states it never saw in the demos ("distribution shift").
  You will see exactly this on PushT.

## 2. The two classic policies we will train (~1–2 h, skim)

- [ ] **Diffusion Policy** (Chi et al., 2023): the paper behind the PushT benchmark
  Paper: https://arxiv.org/abs/2303.04137
  Videos: https://diffusion-policy.cs.columbia.edu/
  Read sections 1–3 and watch the PushT videos. Key idea: generate an action sequence
  by denoising, the same way image diffusion models generate pictures. This handles
  demonstrations where there are several valid ways to do the task ("multimodality").

- [ ] **ACT / ALOHA** (Zhao et al., 2023)
  Paper: https://arxiv.org/abs/2304.13705
  Sections 1 and 4 are enough. Key idea: **action chunking**, which means predicting the
  next k actions at once instead of one per step. This reduces compounding error.

## 3. Vision-language-action (VLA) models (~2–3 h)

- [ ] **RT-2** (Google DeepMind, 2023): the paper that established the idea
  https://arxiv.org/abs/2307.15818
  Introduction and Figure 1. A vision-language model outputs robot actions as text tokens.

- [ ] **OpenVLA** (2024): an open 7B VLA with a good overview of the field
  https://arxiv.org/abs/2406.09246

- [ ] **π0** (Physical Intelligence, 2024): continuous actions via flow matching
  https://arxiv.org/abs/2410.24164
  SmolVLA follows this design.

- [ ] **SmolVLA** (Hugging Face, 2025): the 450M model we will fine-tune. **Read carefully.**
  https://arxiv.org/abs/2506.01844
  Focus on: the frozen vision-language backbone plus a small trainable "action expert",
  flow matching for actions, and asynchronous inference.

## 4. The tool

- [ ] **LeRobot documentation**: https://huggingface.co/docs/lerobot
  Read "Installation"; skim the "SmolVLA" page.

---

**Only one evening?** Read the CS285 lecture, watch the PushT videos on the Diffusion Policy
page, and read the SmolVLA paper. That covers everything in steps 1–3.

## Questions to be able to answer after reading

1. Why does behaviour cloning drift, and how do action chunking and diffusion help?
2. What is "multimodality" in demonstrations, and why does a plain MSE regression policy fail on it?
3. How does a VLA turn a vision-language model into a robot policy? Tokens (RT-2, OpenVLA) vs. a continuous action head (π0, SmolVLA)?
4. Which parts of SmolVLA are frozen during fine-tuning, and why does that matter for an 8 GB GPU?
