// Builds the form from FIELDS, posts it to /api/predict, and renders the result.

const FIELDS = [
  {
    section: 'Patient',
    fields: [
      { key: 'age', label: 'Age', unit: 'years', type: 'number', min: 18, max: 100, step: 1, value: 50 },
      { key: 'sex', label: 'Sex', type: 'segmented', options: [[0, 'Female'], [1, 'Male']], value: 0 },
    ],
  },
  {
    section: 'Vitals',
    fields: [
      { key: 'trestbps', label: 'Resting blood pressure', unit: 'mmHg', type: 'number', min: 80, max: 220, step: 1, value: 130 },
      { key: 'chol', label: 'Cholesterol', unit: 'mg/dL', type: 'number', min: 100, max: 600, step: 1, value: 250 },
      { key: 'fbs', label: 'Fasting blood sugar over 120 mg/dL', type: 'segmented', options: [[0, 'No'], [1, 'Yes']], value: 0 },
    ],
  },
  {
    section: 'Symptoms & ECG',
    fields: [
      { key: 'cp', label: 'Chest pain type', type: 'segmented', options: [[0, '0'], [1, '1'], [2, '2'], [3, '3']], value: 0 },
      { key: 'restecg', label: 'Resting ECG', type: 'segmented', options: [[0, '0'], [1, '1'], [2, '2']], value: 0 },
      { key: 'thal', label: 'Thal', type: 'segmented', options: [[1, '1'], [2, '2'], [3, '3']], value: 1 },
      { key: 'ca', label: 'Major vessels', type: 'segmented', options: [[0, '0'], [1, '1'], [2, '2'], [3, '3']], value: 0 },
    ],
  },
  {
    section: 'Exercise Test',
    fields: [
      { key: 'thalach', label: 'Max heart rate', unit: 'bpm', type: 'number', min: 60, max: 220, step: 1, value: 150 },
      { key: 'oldpeak', label: 'ST depression', unit: 'mm', type: 'number', min: 0, max: 10, step: 0.1, value: 1.0 },
      { key: 'slope', label: 'ST slope', type: 'segmented', options: [[0, '0'], [1, '1'], [2, '2']], value: 0 },
      { key: 'exang', label: 'Exercise-induced angina', type: 'segmented', options: [[0, 'No'], [1, 'Yes']], value: 0 },
    ],
  },
];

const LEVEL_TEXT = {
  low: 'Low risk of heart disease',
  moderate: 'Moderate risk of heart disease',
  high: 'High risk of heart disease',
};

const $ = (id) => document.getElementById(id);
const values = {};

// ---- build the form --------------------------------------------------------
for (const group of FIELDS) {
  const card = document.createElement('fieldset');
  card.className = 'card group';
  card.innerHTML = `<legend>${group.section}</legend>`;
  for (const f of group.fields) {
    values[f.key] = f.value;
    const row = document.createElement('div');
    row.className = 'row';
    const label = document.createElement('label');
    label.textContent = f.label;
    row.append(label);

    if (f.type === 'number') {
      const id = `f-${f.key}`;
      label.htmlFor = id;
      const wrap = document.createElement('div');
      wrap.className = 'num';
      wrap.innerHTML = `
        <button type="button" class="step" data-d="-1" aria-label="Decrease ${f.label}">−</button>
        <input id="${id}" type="number" inputmode="decimal" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.value}">
        <button type="button" class="step" data-d="1" aria-label="Increase ${f.label}">+</button>
        ${f.unit ? `<span class="unit">${f.unit}</span>` : ''}`;
      const input = wrap.querySelector('input');
      const decimals = String(f.step).split('.')[1]?.length || 0;
      const set = (v) => {
        v = Math.min(f.max, Math.max(f.min, v));
        input.value = v.toFixed(decimals);
        values[f.key] = Number(input.value);
        input.classList.remove('invalid');
      };
      input.addEventListener('input', () => {
        const v = Number(input.value);
        const ok = input.value !== '' && v >= f.min && v <= f.max;
        input.classList.toggle('invalid', !ok);
        if (ok) values[f.key] = v;
      });
      input.addEventListener('blur', () => set(input.value === '' ? f.value : Number(input.value)));
      for (const b of wrap.querySelectorAll('.step')) {
        b.addEventListener('click', () => set(values[f.key] + Number(b.dataset.d) * f.step));
      }
      row.append(wrap);
    } else {
      label.id = `l-${f.key}`;
      const seg = document.createElement('div');
      seg.className = 'segmented';
      seg.setAttribute('role', 'radiogroup');
      seg.setAttribute('aria-labelledby', label.id);
      for (const [v, text] of f.options) {
        const b = document.createElement('button');
        b.type = 'button';
        b.textContent = text;
        b.setAttribute('role', 'radio');
        const sync = () => {
          for (const c of seg.children) {
            const on = c === b;
            c.classList.toggle('active', on);
            c.setAttribute('aria-checked', on);
          }
        };
        b.addEventListener('click', () => { values[f.key] = v; sync(); });
        if (v === f.value) queueMicrotask(sync);
        seg.append(b);
      }
      row.append(seg);
    }
    card.append(row);
  }
  $('sections').append(card);
}

// ---- predict ---------------------------------------------------------------
const RING = 2 * Math.PI * 52;
const ring = $('ring-fill');
ring.style.strokeDasharray = RING;
ring.style.strokeDashoffset = RING;

$('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (document.querySelector('.num input.invalid')) {
    showError('One of the numbers is out of range. Check the fields outlined in pink.');
    return;
  }
  const btn = $('predict');
  btn.disabled = true;
  btn.textContent = 'Estimating…';
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(values),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    render(data);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    btn.disabled = false;
    btn.textContent = 'Estimate Risk';
  }
});

function render({ probability, level, recommendations }) {
  $('error').hidden = true;
  const result = $('result');
  result.dataset.level = level;
  ring.style.strokeDashoffset = RING * (1 - probability);
  $('pct').textContent = `${Math.round(probability * 100)}%`;
  $('level').textContent = LEVEL_TEXT[level];
  const list = $('recs');
  list.replaceChildren(...recommendations.map((r) => {
    const li = document.createElement('li');
    li.textContent = r;
    return li;
  }));
  $('recs-wrap').hidden = false;
  if (matchMedia('(max-width: 900px)').matches) result.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showError(msg) {
  const el = $('error');
  el.textContent = msg;
  el.hidden = false;
}
