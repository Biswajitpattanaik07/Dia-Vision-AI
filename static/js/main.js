// DiaVision AI — shared frontend behavior
// Page-specific interactions (drag-and-drop upload, chat, charts) live in
// their respective templates' {% block scripts %}. This file holds
// site-wide helpers: 3D tilt/glow cards and the sliding sidebar indicator.

const PREFERS_REDUCED_MOTION =
  window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

document.addEventListener('DOMContentLoaded', () => {
  // Auto-resize any textarea with the "autosize" data attribute.
  document.querySelectorAll('textarea[data-autosize]').forEach((el) => {
    el.addEventListener('input', function () {
      this.style.height = 'auto';
      this.style.height = this.scrollHeight + 'px';
    });
  });

  initTiltGlowCards();
  initSlidingNavIndicator();
});

/**
 * Cursor-following glow on every glass-card / glass-panel / .tilt-glow
 * element, plus a subtle 3D tilt on SMALL cards only. Large panels (such as
 * the result page's image panel) get the glow but no tilt, because tilting a
 * big panel makes images and text look blurry and distracting.
 * Add data-no-tilt to any element that should never tilt.
 */
function initTiltGlowCards() {
  const targets = document.querySelectorAll('.glass-card, .glass-panel, .tilt-glow');
  const MAX_TILT_WIDTH = 640;   // px - wider than this: glow only
  const MAX_TILT_HEIGHT = 480;  // px - taller than this: glow only

  targets.forEach((el) => {
    el.classList.add('tilt-glow');

    el.addEventListener('mousemove', (e) => {
      const rect = el.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      el.style.setProperty('--mx', `${(x / rect.width) * 100}%`);
      el.style.setProperty('--my', `${(y / rect.height) * 100}%`);

      const canTilt =
        !PREFERS_REDUCED_MOTION &&
        !el.hasAttribute('data-no-tilt') &&
        rect.width <= MAX_TILT_WIDTH &&
        rect.height <= MAX_TILT_HEIGHT;
      if (!canTilt) return;

      // Gentle 3D tilt, clamped to a few degrees so it stays subtle.
      const rotateY = (x / rect.width - 0.5) * 6;
      const rotateX = (y / rect.height - 0.5) * -6;
      el.style.transform =
        `perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(0)`;
    });

    el.addEventListener('mouseleave', () => {
      el.style.transform = '';
    });

    el.addEventListener('click', () => {
      if (PREFERS_REDUCED_MOTION) return;
      el.classList.remove('jump-pop');
      // Force reflow so the animation can re-trigger on repeated clicks.
      void el.offsetWidth;
      el.classList.add('jump-pop');
    });
  });
}

/**
 * Animates a glowing pill that slides to whichever sidebar nav item is
 * hovered, and rests on the active one when the mouse leaves.
 */
function initSlidingNavIndicator() {
  const nav = document.querySelector('aside nav');
  if (!nav) return;

  const links = Array.from(nav.querySelectorAll('.nav-link'));
  if (!links.length) return;

  const slider = document.createElement('div');
  slider.id = 'nav-slider';
  nav.style.position = 'relative';
  nav.appendChild(slider);

  const active = nav.querySelector('.nav-link.active');

  function moveSliderTo(el) {
    slider.style.top = `${el.offsetTop}px`;
    slider.style.height = `${el.offsetHeight}px`;
    slider.classList.add('visible');
  }

  function restSlider() {
    if (active) {
      moveSliderTo(active);
    } else {
      slider.classList.remove('visible');
    }
  }

  restSlider();

  links.forEach((link) => {
    link.addEventListener('mouseenter', () => moveSliderTo(link));
  });
  nav.addEventListener('mouseleave', restSlider);

  // Keep the pill aligned if the window is resized or fonts finish loading.
  window.addEventListener('resize', restSlider);
  window.addEventListener('load', restSlider);
}