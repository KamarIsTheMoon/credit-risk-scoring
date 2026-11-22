# Results

Data: 1000 borrowers, 30% defaulted. 5-fold cross-validation.

| Model | AUC | Best refusal threshold | Refusal rate | Defaults caught | Cost per applicant | Cost if we approve everyone |
|---|---|---|---|---|---|---|
| Logistic regression | 0.740 | P(default) >= 0.18 | 61% | 86% | 0.56 | 1.50 |
| Gradient boosting | 0.748 | P(default) >= 0.12 | 71% | 89% | 0.60 | 1.50 |

## Strongest drivers in the logistic model (positive = more risk)

| feature                    |   effect_on_default |
|:---------------------------|--------------------:|
| checking_account_not_known |              -1.002 |
| purpose_education          |               0.809 |
| checking_account_little    |               0.677 |
| saving_accounts_little     |               0.547 |
| purpose_vacation/others    |              -0.543 |
| saving_accounts_not_known  |              -0.477 |
| saving_accounts_rich       |              -0.473 |
| duration                   |               0.418 |

## Input importance in the gradient boosting model

| feature          |   auc_drop |
|:-----------------|-----------:|
| checking_account |     0.1122 |
| duration         |     0.0516 |
| credit_amount    |     0.0505 |
| saving_accounts  |     0.0164 |
| age              |     0.0078 |
| job              |     0.0003 |
| purpose          |    -0.0002 |
| housing          |    -0.0026 |

## Example: why the riskiest test applicant (P(default) = 90%) is refused

|                           |   effect_on_default |
|:--------------------------|--------------------:|
| duration                  |               1.381 |
| purpose_education         |               0.809 |
| saving_accounts_little    |               0.547 |
| checking_account_moderate |               0.377 |

## Fairness check: approval rate by sex (not an input of the model)

| sex    | approval_rate   |
|:-------|:----------------|
| female | 37%             |
| male   | 40%             |
