# Data source

Hofmann, H. (1994). *Statlog (German Credit Data)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77

This project uses the 10-column version of the dataset that ships with the dalex Python package (`dalex.datasets.load_german()`): 1,000 borrowers, 300 of whom defaulted.

Columns: risk (1 good, 0 bad), sex, job, housing, saving_accounts, checking_account, credit_amount, duration (months), purpose, age.

`german_credit.csv` is created automatically the first time `python -m src.run` is run.
