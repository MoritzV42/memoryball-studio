(() => {
  const timeline = document.querySelector('.timeline');
  if (!timeline) {
    return;
  }

  const track = timeline.querySelector('.timeline__track');
  const handle = timeline.querySelector('.timeline__handle');
  const progress = timeline.querySelector('.timeline__progress');
  const labels = Array.from(timeline.querySelectorAll('.timeline__label'));
  const cards = Array.from(document.querySelectorAll('.process-card'));
  const steps = cards.length;

  let orientation = 'horizontal';
  let trackRect = track.getBoundingClientRect();
  let dragging = false;
  let currentStep = 0;

  const clamp01 = (value) => Math.min(1, Math.max(0, value));
  const fractionForStep = (step) => (steps > 1 ? step / (steps - 1) : 0);

  function updateOrientation() {
    const prefersVertical = window.matchMedia('(max-width: 720px)').matches;
    orientation = prefersVertical ? 'vertical' : 'horizontal';
    timeline.dataset.orientation = orientation;
    trackRect = track.getBoundingClientRect();
    setHandleToStep(currentStep);
  }

  function setActiveStep(step) {
    currentStep = step;
    cards.forEach((card) => {
      card.classList.toggle('is-active', Number(card.dataset.step) === step);
    });
    labels.forEach((label) => {
      label.classList.toggle('is-active', Number(label.dataset.step) === step);
    });
  }

  function positionHandle(fraction) {
    trackRect = track.getBoundingClientRect();
    const handleSize = orientation === 'horizontal' ? handle.offsetWidth : handle.offsetHeight;

    if (orientation === 'horizontal') {
      const usable = Math.max(trackRect.width - handleSize, 0);
      const center = handleSize / 2 + usable * fraction;
      handle.style.left = `${center}px`;
      handle.style.top = '50%';
      progress.style.height = '';
      progress.style.width = `${usable * fraction}px`;
    } else {
      const usable = Math.max(trackRect.height - handleSize, 0);
      const center = handleSize / 2 + usable * fraction;
      handle.style.top = `${center}px`;
      handle.style.left = '50%';
      progress.style.width = '';
      progress.style.height = `${usable * fraction}px`;
    }
  }

  function setHandleToStep(step) {
    const fraction = fractionForStep(step);
    positionHandle(fraction);
    setActiveStep(step);
  }

  function fractionFromEvent(event) {
    const pointX = event.clientX ?? (event.touches ? event.touches[0].clientX : 0);
    const pointY = event.clientY ?? (event.touches ? event.touches[0].clientY : 0);
    if (orientation === 'horizontal') {
      const fraction = (pointX - trackRect.left) / trackRect.width;
      return clamp01(fraction || 0);
    }
    const fraction = (pointY - trackRect.top) / trackRect.height;
    return clamp01(fraction || 0);
  }

  function updateFromEvent(event) {
    event.preventDefault();
    trackRect = track.getBoundingClientRect();
    const fraction = fractionFromEvent(event);
    const step = Math.round(fraction * (steps - 1));
    positionHandle(fraction);
    if (step !== currentStep) {
      setActiveStep(step);
    }
  }

  function endDrag(event) {
    if (!dragging) {
      return;
    }
    if (event) {
      event.preventDefault();
    }
    dragging = false;
    if (event && track.hasPointerCapture(event.pointerId)) {
      track.releasePointerCapture(event.pointerId);
    }
    const snapFraction = fractionForStep(currentStep);
    positionHandle(snapFraction);
  }

  track.addEventListener('pointerdown', (event) => {
    dragging = true;
    trackRect = track.getBoundingClientRect();
    track.setPointerCapture(event.pointerId);
    updateFromEvent(event);
  });

  track.addEventListener('pointermove', (event) => {
    if (!dragging) {
      return;
    }
    updateFromEvent(event);
  });

  const stopDrag = (event) => {
    if (!dragging) {
      return;
    }
    endDrag(event);
  };

  track.addEventListener('pointerup', stopDrag);
  track.addEventListener('pointercancel', stopDrag);
  track.addEventListener('lostpointercapture', stopDrag);

  labels.forEach((label) => {
    label.addEventListener('click', () => {
      const step = Number(label.dataset.step);
      setHandleToStep(step);
    });
  });

  window.addEventListener('resize', () => {
    updateOrientation();
  });

  updateOrientation();
  setHandleToStep(0);
})();
