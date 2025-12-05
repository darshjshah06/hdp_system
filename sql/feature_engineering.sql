TRUNCATE features RESTART IDENTITY;

INSERT INTO features (
    id,
    age,
    age_group,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal,
    target
)
SELECT
    id,
    age,

    -- Age groups
    CASE
        WHEN age < 40 THEN 'under_40'
        WHEN age BETWEEN 40 AND 54 THEN '40_54'
        WHEN age BETWEEN 55 AND 69 THEN '55_69'
        ELSE '70_plus'
    END AS age_group,

    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal,
    target
FROM patients_cleaned;
