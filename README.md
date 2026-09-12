# Why Do I Keep Losing?

> *"We lost because of the draft."*  
> *"My teammates are terrible."*  
> *"This hero is broken."*

These are things every Dota 2 player has said at some point.

Dota 2 is an incredibly complex game with countless variables that can influence whether you get the **+25** or the **-25**. Hero composition, player skill, item choices, individual performance, and a bit of questionable decision-making can all play a role.

But how much of what we believe about Dota is actually true?

And, more importantly, **can we use AI and data to find out?**

Either way, you're probably still going to queue next.

---

## Purpose

**Why Do I Keep Losing?** is a casual machine learning and data analysis project for testing common theories and beliefs about Dota 2.

The goal isn't to build the world's greatest Dota 2 prediction model. Instead, this project is about taking questions that Dota players commonly ask and seeing whether the data actually supports them.

Some examples:

- **"We lost in the draft."**
- Does hero composition actually have a significant impact on the outcome?
- Are some heroes stronger together than their individual win rates suggest?
- Can a model predict the outcome from the draft alone?
- What other common Dota theories can we test?

The results might confirm a theory, debunk it, or simply raise more questions.

Most importantly, **this is meant to be fun.**

If you have a Dota theory you'd like to investigate, feel free to open an issue, experiment with the notebooks, or contribute your own analysis.

---

## Current Investigations

### 01 — We Lost in the Draft

The first investigation looks at whether a Dota 2 match outcome can be predicted from the **hero draft alone**.

The project experiments with representing hero drafts as images and training image classification models to distinguish between Radiant wins and Radiant losses.

Models currently explored include:

- ResNet
- Vision Transformers (ViT)
- Pretrained and randomly initialized models
- Different hero ordering strategies

The full analysis and results can be found in:

`notebooks/01_HeroPicViT_Example.ipynb`

and

`notebooks/02_Hero_Pic_ViT_Model_Training.ipynb`

A write-up of the investigation is also available in:

`articles/01_hero_pic_vit_article.pdf`

---

## Repository Structure

```text
.
├── articles/       # Write-ups and project articles
├── notebooks/      # Experiments and analysis
├── scripts/        # Dataset download, extraction, and utility scripts
├── src/            # Reusable Python modules
└── tests/          # Unit tests

## Getting Started

Clone the repository:

```bash
git clone <repository-url>
cd why-do-I-keep-losing
```

Install the project and its dependencies:

```bash
pip install -e .
```

From there, the notebooks in `notebooks/` can be used to reproduce the experiments.

## Contributing

Have a Dota theory you want to test?

I'd love to hear it.

You can contribute by:

- Suggesting a theory to investigate
- Opening an issue with a research question
- Improving the dataset or analysis
- Adding a new model
- Reproducing an experiment
- Finding something interesting in the data

You don't need to be an AI/ML expert. If there's a Dota question you're curious about, that's enough.

## Disclaimer

This project is a casual experiment, not a serious attempt to solve Dota 2.

Machine learning models can find patterns in data, but that doesn't necessarily mean those patterns represent causation or explain why a match was won or lost.

So if the model says your draft was terrible...

Maybe it really was your team's fault after all.
