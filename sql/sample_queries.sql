-- Count patients with heart disease
SELECT COUNT(*) FROM features WHERE target = 1;

-- Average cholesterol by age group
SELECT age_group, AVG(chol) AS avg_chol
FROM features
GROUP BY age_group
ORDER BY avg_chol DESC;
