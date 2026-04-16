const analyzeBtn = document.getElementById('analyzeBtn');
const inputText = document.getElementById('inputText');

const resultsEl = document.getElementById('results');
const scoreValueEl = document.getElementById('scoreValue');
const riskLabelEl = document.getElementById('riskLabel');
const riskBarEl = document.getElementById('riskBar');
const featureTableEl = document.getElementById('featureTable');
const explanationsEl = document.getElementById('explanations');
const headlineEl = document.getElementById('headline');
const sentenceAnalysisEl = document.getElementById('sentenceAnalysis');
const paragraphAnalysisEl = document.getElementById('paragraphAnalysis');
const warningsEl = document.getElementById('warnings');

const renderWarnings = (warnings = []) => {
  warningsEl.innerHTML = '';
  warnings.forEach(w => {
    const p = document.createElement('p');
    p.className = 'warning';
    p.textContent = w;
    warningsEl.appendChild(p);
  });
};

analyzeBtn.addEventListener('click', async () => {
  renderWarnings([]);
  const text = inputText.value || '';

  const response = await fetch('/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });

  const data = await response.json();
  if (!data.ok) {
    renderWarnings(data.errors || ['Error inesperado']);
    return;
  }

  const result = data.result;
  renderWarnings(data.warnings || []);

  scoreValueEl.textContent = result.score.score_0_100;
  riskLabelEl.textContent = `Riesgo: ${result.score.risk_level} | Modelo: ${result.score.model_name}`;
  riskBarEl.style.width = `${result.score.score_0_100}%`;

  headlineEl.textContent = result.explanation.headline;
  explanationsEl.innerHTML = '';
  result.explanation.details.forEach(d => {
    const li = document.createElement('li');
    li.textContent = d;
    explanationsEl.appendChild(li);
  });

  featureTableEl.innerHTML = '';
  Object.entries(result.features).forEach(([k, v]) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${k}</td><td>${Number(v).toFixed(4)}</td>`;
    featureTableEl.appendChild(tr);
  });

  sentenceAnalysisEl.innerHTML = result.sentence_analysis.map((s, idx) => `
    <article><strong>Oración ${idx + 1}:</strong> ${s.sentence}<br>
    Palabras: ${s.word_count}, Densidad puntuación: ${s.punctuation_density.toFixed(4)},
    Conectores: ${s.connector_density.toFixed(4)}</article><hr>
  `).join('');

  paragraphAnalysisEl.innerHTML = result.paragraph_analysis.map((p, idx) => `
    <article><strong>Párrafo ${idx + 1}:</strong> palabras=${p.word_count}, oraciones=${p.sentence_count}</article><hr>
  `).join('');

  resultsEl.classList.remove('hidden');
});
