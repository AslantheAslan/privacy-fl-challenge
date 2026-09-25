# Synthetic Dataset Card

## Intended use

This dataset is designed solely as a recruitment and educational benchmark for clinical NLP, privacy-preserving machine learning, and cross-silo federated learning.

## Composition

The complete benchmark contains 200 generated cases:

- training: 120;
- public validation: 30;
- private hidden test: 50.

Three synthetic hospital nodes differ in note format, terminology, units, demographic distributions, disease frequencies, and generated outcome risk. The hidden split includes additional formatting variants.

## Generation

Clinical conditions, medications, measurements, missingness, structured features, PII, and outcomes are sampled from deterministic programmatic templates and probability distributions. The outcome depends probabilistically on selected risk factors and a site effect, with additional random noise.

All names, addresses, telephone numbers, IDs, emails and clinical narratives are generated. Accidental resemblance to real persons is coincidental. The `.example` email domain is reserved for documentation.

## Deliberate challenge properties

- cross-site non-IID distributions;
- abbreviations, synonyms and medication brands;
- metric-unit conversion;
- decimal commas;
- missing values;
- negated and family-history distractors;
- repeated PII entities;
- formatting shift in the hidden split.

## Limitations

The data are templated and substantially simpler than real-world longitudinal medical records. They do not represent clinical validity, epidemiology, actual institutional workflows, or the full complexity of multilingual and multimodal de-identification. Benchmark success must not be interpreted as evidence that a system is safe for clinical deployment.
