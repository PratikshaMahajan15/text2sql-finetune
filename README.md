# text2sql-finetune

Fine-tuning Qwen2.5-1.5B-Instruct with QLoRA to write SQL from a table schema and a plain-English question. I built this to go through the full cycle myself: data preparation, training, measuring before and after, and publishing the result.

Model on Hugging Face: [PratikshaMahajan/qwen2.5-1.5b-text2sql-qlora](https://huggingface.co/PratikshaMahajan/qwen2.5-1.5b-text2sql-qlora)

## Results

Tested on 200 questions that were not used in training. A prediction counts as correct if it matches the reference SQL after light normalization (lowercase, extra whitespace removed, double quotes turned into single quotes, markdown fences and trailing semicolons stripped).

| Model | Exact-match accuracy |
|---|---|
| Qwen2.5-1.5B-Instruct (base) | 46.0% |
| Base + LoRA adapter | 68.0% |

Example from the test set:

```
Question: How many Goals against have Played more than 34?
Reference:  SELECT COUNT(goals_against) FROM table_name_97 WHERE played > 34
Base model: ```sql SELECT COUNT(goals_against) FROM table_name_97 WHERE played > 34; ```
Fine-tuned: SELECT COUNT(goals_against) FROM table_name_97 WHERE played > 34
```
## Demo

I ran the published adapter in a Gradio app on a free Colab GPU and tried a few inputs of my own. These are screenshots from that session, not a measured result. The accuracy figure above comes from the 200-question test set.
**Aggregate on a new schema**
<img width="1920" height="988" alt="salary" src="https://github.com/user-attachments/assets/96a0bf8e-191b-486f-925d-44dfeca99b7f" />

**Filter on a text column**
<img width="1920" height="990" alt="demo-pune" src="https://github.com/user-attachments/assets/bb1f837d-3217-45f2-b9d6-a863727dd1e6" />

**A simple two-table join.** The query joins on the right columns, but it has no `DISTINCT`, so a customer with several orders would be listed several times. I only tried this one join.
<img width="1920" height="979" alt="demo-join" src="https://github.com/user-attachments/assets/be2d6d43-fc1d-4dc1-b45e-440eb6fd0946" />

**Wrong aggregation.** For "How many heads are older than 56?" the model returned `SUM(age)`. The correct query is `SELECT COUNT(*) FROM head WHERE age > 56`.
<img width="1920" height="983" alt="demo-failure-sum" src="https://github.com/user-attachments/assets/8861ce96-0b70-4bbb-b2c8-d5fcb7598f4c" />

# Setup

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`, loaded in 4-bit (NF4)
- Data: 2,000 random rows from [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) (seed 42), split into 1,800 train and 200 test
- LoRA: r=16, alpha=32, dropout 0.05, all linear layers
- Training: 2 epochs, learning rate 2e-4 (cosine), batch size 4 with 4 gradient accumulation steps, max length 512, fp16
- Hardware: one Tesla T4 on Kaggle, about 8.5 minutes
- Libraries: `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`

# Files

- `text2sql_qlora.ipynb`: the full notebook (data, baseline, training, evaluation, upload)

# Things I ran into

- `bitsandbytes` was not picked up until I restarted the session after installing it.
- `SFTTrainer` refused to run with two GPUs visible, so I limited it to one with `CUDA_VISIBLE_DEVICES=0`.
- Training failed with a bfloat16 vs fp16 mismatch on the T4. Loading the model in float16 and keeping the LoRA weights in float32 fixed it.
- I measured the base model by switching the adapter off (`disable_adapter()`) instead of reloading the model.

# Limitations

- Exact match counts a correct query written differently (another column order, an alias) as wrong. I did not execute the queries against a database, so real accuracy is probably higher.
- The training data is mostly single-table questions with short schemas, so joins and nested queries are weak.
- The test set is only 200 questions, and I have not tried it on real company schemas.

# What I would try next

- Evaluate by executing queries and comparing results
- Train on Spider for multi-table questions
- Compare against a prompt-engineered base model
