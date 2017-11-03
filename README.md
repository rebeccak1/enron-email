# enron-email

The purpose of this project is to predict whether an individual is a person of interest. The
dataset includes 18 POI and 128 non-POI, for a total number of 146 data points. For each
individual, the dataset provides these 21 features:

a. financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances',
'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
'director_fees'] (all units are in US dollars)
b. email features: ['to_messages', 'email_address', 'from_poi_to_this_person',
'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are
generally number of emails messages; notable exception is ‘email_address’, which is a
text string)
c. POI label: [‘poi’] (boolean, represented as integer)

The report for the project is at `enron_report.pdf`.

This project is part of the Udacity Data Analyst Nanodegree.
