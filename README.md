# GenAI Workload Power Prediction and Energy Aware Infrastructure Optimization

AAI-540 · Group 6

Generative AI workloads are growing quickly, and their power draw varies with workload type and configuration. This project predicts the mean power of a GenAI inference execution from its configuration *before it runs*, and uses those predictions in a simulated power aware scheduler for capacity planning. The ML workflow is also implemented on AWS SageMaker.

All scheduling results are **simulations** on historical measurements. They do not represent real infrastructure control or measured savings.

## Dataset

[NLR Generative AI Workload Power Profiles Dataset](https://data.nlr.gov): power measurements from GenAI workloads on an HPC platform. See the dataset's own README for the full citation (Vercellino et al.).

| Workload group | Executions used |
|---|---|
| Llama3 70B offline inference | 1,200 |
| Llama3 70B online finite-rate inference | 1,024 (2 incomplete runs removed) |
| Llama3 70B online rate based inference | 200 |
| Llama2 70B LoRA / Stable Diffusion training | 41 (kept separate, too few for modeling) |

The modeling dataset has **2,424 inference executions**. The target is `mean_power_w`. The six predictors, all known before execution, are `workload_type`, `batch_size`, `max_output_tokens`, `request_rate`, `output_length` and `dataset`. Peak power, duration, throughput, latency, completion status and identifiers are excluded to avoid leakage. `num_prompts` is excluded because it is constant or perfectly collinear with `request_rate`.

## Repository structure

```
notebooks/
├── 01_EDA_V1.ipynb                                  Dataset exploration and modeling dataset
├── 02_Preprocessing_V1.ipynb                        Leakage audit, split, preprocessing pipeline
├── 03_Modeling .ipynb                               Five models, selection, explainability
├── 04_Infrastructure_Optimization.ipynb             Simulated power aware scheduling
├── AAI-540_Group6_SageMaker_5_Models-70-15-15.ipynb SageMaker workflow, 70/15/15 split
├── AAI-540_Group6_SageMaker_5_Models_80_20.ipynb    SageMaker workflow, 80/20 split variant
├── Latest_Project_File.ipynb                        
└── Latest_Project_File_scheduler.ipynb              Simulated power aware scheduling SageMakerVersion
README.md
```

## Workflow

```
Workload request 
        │
        ▼
Power prediction model 
        │
        ▼
Power aware scheduler  ◄── available slots, power budget
        │
        ▼
Scheduling decision → simulated execution → observed power → evaluation
```

1. **EDA** verifies that Parquet power traces match metadata records, shows that power rises with batch size and request rate before levelling off, removes two incomplete runs, and saves a clean `inference_modeling_dataset.csv` (2,424 rows × 7 columns).
2. **Preprocessing** applies a stratified 70/15/15 split by workload type (1,696 / 364 / 364 rows, `random_state=42`) *before* any fitting. Numeric features get median imputation, missing-value indicators and standard scaling. Categorical features get one hot encoding. The pipeline is fitted on training data only and produces 14 features, saved with `preprocessor.joblib`.
3. **Modeling** compares five models on the validation set, then evaluates the selected model once on the test set.
4. **Infrastructure optimization** loads the saved model and compares a first-come-first-served baseline with a greedy, lowest-predicted-power-first scheduler that keeps a safety margin for prediction error. It uses 4 execution slots and hypothetical budgets of 8–12 kW. Actual power is used only after scheduling, to evaluate decisions.
5. **SageMaker** notebooks replicate the workflow with Feature Store (online and offline stores), SageMaker training jobs for the five models, and endpoint deployment.

## Running the notebooks

The Colab notebooks (01–04) are meant to be run in order, with Google Drive mounted. They exchange files under `/content/drive/MyDrive/data/processed/`:

| Notebook | Reads | Writes |
|---|---|---|
| 01 EDA | `/content/dataset.zip` | `inference_modeling_dataset.csv` |
| 02 Preprocessing | `inference_modeling_dataset.csv` | `modeling/` (processed splits, `preprocessor.joblib`) |
| 03 Modeling | `modeling/` | `modeling/best_model.joblib` |
| 04 Infrastructure Optimization | `modeling/` | `optimization/` (predictions, config, scheduler) |

The dataset ZIP must be uploaded to `/content` before running notebook 01. The SageMaker notebooks need an AWS account with SageMaker access and `inference_modeling_dataset.csv` in the working directory.

**Requirements:** Python 3, pandas, numpy, scikit-learn, xgboost, joblib, matplotlib, shap (optional). The SageMaker notebooks use the SageMaker Python SDK v2 and boto3. Training and inference should use the same scikit-learn version, because the preprocessor and Random Forest models are pickled.

## Limitations

- Scheduling results are simulations with hypothetical budgets, not measured infrastructure savings.
- The scheduler is a simple greedy heuristic, not an optimal policy.
- Power is predicted per execution (mean power), not as a time series.
- Only 41 training executions exist, so training workloads are not modeled.

## Future Work

### Composition Aware Uncertainty Reservation for Concurrent GenAI Workload Scheduling

The current scheduler uses a safety reserve based on the number of concurrently scheduled workloads. While this provides a simple buffer against prediction errors, it does not account for differences in workload composition.

A future extension is to develop a **composition aware uncertainty reservation model** that dynamically predicts the required power safety margin based on the specific workloads scheduled together.

Unlike a static lookup table or job count based reserve, the proposed model would consider:

- **Workload composition:** The combination of workload types running concurrently.
- **Configuration characteristics:** Batch size, request rate, output length, and other relevant workload parameters.
- **Predicted power distribution:** Individual power estimates, prediction uncertainty, and variation across selected workloads.
- **Concurrency effects:** How combined workload characteristics influence aggregate power prediction errors.

The predicted uncertainty margin would be incorporated into the scheduler's admission decisions to reduce the risk of exceeding the available power budget while avoiding unnecessarily conservative reservations.

**Preliminary simulation observations suggest** that composition aware reserves may provide better power budget protection than count only pooling while requiring comparable or less reserved headroom. These observations require further validation across diverse workload combinations and operating conditions.

The long term objective is to develop and evaluate a **Composition Aware Uncertainty Reservation Model for Concurrent GenAI Workload Scheduling**, enabling more adaptive, reliable, and power efficient infrastructure capacity planning.

