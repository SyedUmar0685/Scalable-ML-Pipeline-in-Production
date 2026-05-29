# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier trained using scikit-learn with 100 estimators and a fixed random state of 42. It predicts whether an individual's income exceeds $50K/year based on census data. The model was developed as part of the Udacity MLOps Nanodegree project.

## Intended Use

This model is intended for educational and demonstration purposes to classify individuals into two income brackets (<=50K or >50K) based on demographic and employment features. It should not be used for making real-world decisions about individuals.

## Training Data

The training data is derived from the 1994 Census Bureau database (also known as the "Adult" dataset). The cleaned dataset contains 32,561 rows and 15 columns. An 80/20 train-test split was used with a random state of 42. Categorical features (workclass, education, marital-status, occupation, relationship, race, sex, native-country) were one-hot encoded, and the label (salary) was binarized.

## Evaluation Data

The evaluation data comprises 20% of the total cleaned census dataset, held out using stratified splitting with random_state=42. The same preprocessing pipeline (encoder and label binarizer) fitted on the training data was applied to the test set.

## Metrics

The model was evaluated using Precision, Recall, and F-beta (F1) score:
- **Precision**: 0.7419 — Of all predicted >50K, approximately 74% actually earned >50K.
- **Recall**: 0.6384 — Of all actual >50K earners, approximately 64% were correctly identified.
- **F-beta (F1)**: 0.6863 — The harmonic mean of precision and recall.

Slice-based performance metrics on categorical features are documented in `slice_output.txt`.

## Ethical Considerations

The dataset contains sensitive demographic attributes including race, sex, and national origin. The model may exhibit bias along these dimensions due to historical inequalities reflected in the training data. Users should be cautious about deploying this model in any context where decisions could disproportionately affect protected groups. Regular fairness audits using the slice-based metrics are recommended.

## Caveats and Recommendations

- The model is trained on data from 1994, which may not reflect current income distributions or demographic patterns.
- Performance varies across demographic slices; some minority groups have significantly fewer training samples, leading to less reliable predictions.
- This model should not be used as the sole basis for any consequential decision-making.
- For production use, consider retraining on more recent data, applying fairness constraints, and conducting thorough bias testing.
