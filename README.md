# GNR Project – Deep Learning MCQ Solver

Solves deep learning multiple-choice questions from PNG images using **Qwen2-VL-7B-Instruct**, a vision-language model with strong reasoning on code and mathematical content.

---

## Files

| File | Purpose |
|---|---|
| `inference.py` | Main inference script (accepts `--test_dir`) |
| `setup.bash` | Environment + model setup (run with internet) |
| `requirements.txt` | Python dependencies |
| `kaggle_test_notebook.ipynb` | Notebook for testing on Kaggle T4x2 |

---

## Setup & Run (Grading System)

The grader runs these commands exactly:

```bash
# 1 — Setup (internet available)
bash setup.bash

# 2 — Activate environment
conda activate gnr_project_env

# 3 — Run inference (no internet)
python inference.py --test_dir /absolute/path/to/test_dir

# 4 — Grade
python grading_script.py --submission_file submission.csv
```

`submission.csv` is created in the **current working directory**.

---

## Environment

| Setting | Value |
|---|---|
| Conda env name | `gnr_project_env` |
| Python version | `3.11` |
| Model | Qwen2-VL-7B-Instruct |
| Model location (after setup) | `~/models/Qwen2-VL-7B-Instruct` |

---

## Kaggle Testing (T4 x2)

1. Open `kaggle_test_notebook.ipynb` on Kaggle
2. **Add model**: Right panel → Add data → search `Qwen2-VL-7B-Instruct` → add
3. **Add test data**: Add your dataset containing `test.csv` + `images/`
4. Set accelerator to **GPU T4 x2**
5. Update `TEST_DIR` in the notebook cell to your dataset path
6. Run all cells → download `submission.csv`

---

## Scoring Logic

| Outcome | Score |
|---|---|
| Correct (1/2/3/4) | +1 |
| Incorrect (1/2/3/4) | -0.25 |
| Unanswered (5) | 0 |
| Hallucinated (anything else) | -1 |

The model uses chain-of-thought reasoning before giving the final answer, which significantly improves accuracy. It outputs 5 only when it genuinely cannot determine the answer.

---

## Model Details

- **Qwen2-VL-7B-Instruct**: 7B-parameter vision-language model
- Understands LaTeX math, code blocks, and technical diagrams in images
- Loaded in `bfloat16` on L40s (48GB) for full precision
- Loaded in `4-bit NF4` quantization on T4 (16GB) for memory efficiency
- Typical speed: ~10–20 seconds per question on L40s

---

## References

- [Qwen2-VL Model Card](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [Transformers Library](https://github.com/huggingface/transformers)
- [BitsAndBytes Quantization](https://github.com/TimDettmers/bitsandbytes)
