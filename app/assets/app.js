const ui = new WebUI();
const $ = (s) => document.querySelector(s);
let labelsBuilt = false;

ui.on_connect(() => { $('#conn').textContent = 'connected'; $('#conn').className = 'pill on'; });
ui.on_disconnect(() => { $('#conn').textContent = 'disconnected'; $('#conn').className = 'pill off'; });

$('#modes').addEventListener('click', (e) => {
  const mode = e.target.closest('button')?.dataset.mode;
  if (mode) ui.send_message('set_mode', { mode });
});

$('#labels').addEventListener('click', (e) => {
  const label = e.target.closest('button')?.dataset.label;
  if (label) ui.send_message('set_label', { label });
});

$('#recordings').addEventListener('click', (e) => {
  const name = e.target.closest('button')?.dataset.delete;
  if (name && confirm(`Delete ${name}?`)) ui.send_message('delete_recording', { name });
});

function buildLabels(labels) {
  $('#labels').innerHTML = labels
    .map((l) => `<button data-label="${l}">${l[0].toUpperCase() + l.slice(1)}</button>`)
    .join('');
  labelsBuilt = true;
}

function select(container, attr, value) {
  container.querySelectorAll('button').forEach((b) => b.classList.toggle('sel', b.dataset[attr] === value));
}

ui.on_message('status', (s) => {
  if (!labelsBuilt) buildLabels(s.labels);
  select($('#modes'), 'mode', s.mode);
  select($('#labels'), 'label', s.label);
  document.body.dataset.mode = s.mode;

  $('#live-record').hidden = s.mode !== 'record';
  $('#live-classify').hidden = s.mode !== 'classify';
  $('#live-idle').hidden = s.mode !== 'idle';
  $('#rate').textContent = s.rate_hz;
  $('#error').hidden = !s.error;
  $('#error').textContent = s.error || '';

  if (s.mode === 'record') {
    $('#rec-seconds').textContent = Math.floor(s.seconds);
    $('#rec-label').textContent = s.label;
    $('#rec-file').textContent = s.file;
  }
  if (s.mode === 'classify') {
    $('#dir').textContent = s.direction;
    const p = s.prediction;
    $('#m1').textContent = p ? p.model1 : 'warming up…';
    $('#m2').textContent = p ? p.model2 : 'warming up…';
    $('#peak').textContent = p ? p.peak_g.toFixed(2) : '–';
  }

  const totals = {};
  s.recordings.forEach((r) => {
    const l = r.name.split('_')[0];
    totals[l] = (totals[l] || 0) + 1;
  });
  $('#rec-summary').textContent = s.labels.map((l) => `${l} ${totals[l] || 0}`).join(' · ');
  $('#recordings').innerHTML = s.recordings.length
    ? s.recordings
        .map((r) => {
          const live = s.mode === 'record' && r.name === s.file;
          return `<li><code>${r.name}</code><span>${live ? 'recording…' : r.seconds + ' s'}</span>` +
            (live ? '' : `<button class="del" data-delete="${r.name}" title="Delete">✕</button>`) + '</li>';
        })
        .join('')
    : '<li class="dim">None yet</li>';
});
