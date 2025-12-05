-- CLEAN TABLE BEFORE INSERTING
TRUNCATE patients_cleaned RESTART IDENTITY;

INSERT INTO patients_cleaned (
    id,
    age,
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
    COALESCE(ca, 0) AS ca,
    COALESCE(thal, 2) AS thal,
    target
FROM patients_raw
WHERE age IS NOT NULL
  AND sex IS NOT NULL
  AND cp IS NOT NULL
  AND trestbps IS NOT NULL
  AND chol IS NOT NULL
  AND fbs IS NOT NULL
  AND restecg IS NOT NULL
  AND thalach IS NOT NULL
  AND exang IS NOT NULL
  AND oldpeak IS NOT NULL
  AND slope IS NOT NULL
  AND target IS NOT NULL;
