// ============================================
// Mobile Navigation
// ============================================
const hamburger = document.getElementById('hamburger');
const navMenu = document.getElementById('nav-menu');

hamburger.addEventListener('click', () => {
    const isOpen = hamburger.classList.toggle('active');
    navMenu.classList.toggle('active');
    hamburger.setAttribute('aria-expanded', isOpen);
});

// Close mobile menu on link click
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
        hamburger.classList.remove('active');
        navMenu.classList.remove('active');
        hamburger.setAttribute('aria-expanded', 'false');
    });
});

// ============================================
// Smooth Scrolling
// ============================================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});

// ============================================
// Navbar Scroll Effect & Active Link
// ============================================
const navbar = document.querySelector('.navbar');
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-link');

window.addEventListener('scroll', () => {
    // Navbar shadow
    navbar.classList.toggle('scrolled', window.scrollY > 30);

    // Active nav link
    let current = '';
    sections.forEach(section => {
        const top = section.offsetTop - 100;
        if (window.scrollY >= top) {
            current = section.getAttribute('id');
        }
    });
    navLinks.forEach(link => {
        link.classList.toggle('active', link.getAttribute('href') === '#' + current);
    });
}, { passive: true });

// ============================================
// Typing Animation
// ============================================
const typedTextSpan = document.querySelector('.typed-text');
const cursorSpan = document.querySelector('.cursor');

const textArray = [
    'Computational Physicist',
    'Numerical Methods Developer',
    'ML Researcher',
    'Quant Modeller',
    'Scientific Computing Engineer'
];
const typingDelay = 90;
const erasingDelay = 50;
const newTextDelay = 2000;
let textArrayIndex = 0;
let charIndex = 0;

function typeChar() {
    if (charIndex < textArray[textArrayIndex].length) {
        cursorSpan.classList.add('typing');
        typedTextSpan.textContent += textArray[textArrayIndex].charAt(charIndex);
        charIndex++;
        setTimeout(typeChar, typingDelay);
    } else {
        cursorSpan.classList.remove('typing');
        setTimeout(eraseChar, newTextDelay);
    }
}

function eraseChar() {
    if (charIndex > 0) {
        cursorSpan.classList.add('typing');
        typedTextSpan.textContent = textArray[textArrayIndex].substring(0, charIndex - 1);
        charIndex--;
        setTimeout(eraseChar, erasingDelay);
    } else {
        cursorSpan.classList.remove('typing');
        textArrayIndex = (textArrayIndex + 1) % textArray.length;
        setTimeout(typeChar, typingDelay + 500);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    if (textArray.length) setTimeout(typeChar, newTextDelay + 250);
});

// ============================================
// Terminal Output Animation (Hero)
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    const output = document.getElementById('terminal-output');
    if (!output) return;

    const lines = [
        { type: 'prompt', text: '→ Running geodesic_tracer.py' },
        { type: 'dim', text: '  Integrating null geodesics (RK4) ...' },
        { type: 'success', text: '✓ Photon ring converged at r = 3M' },
        { type: 'blank' },
        { type: 'prompt', text: '→ Running nbody_sim.py' },
        { type: 'dim', text: '  Verlet integration: N=1000, dt=0.01' },
        { type: 'success', text: '✓ Energy drift: ΔE/E₀ < 10⁻⁸' },
        { type: 'blank' },
        { type: 'prompt', text: '→ Running pinn_solver.py' },
        { type: 'dim', text: '  Training PINN: epoch 5000/5000' },
        { type: 'success', text: '✓ PDE residual: ‖ℒ[u_θ]‖ → 0' },
        { type: 'cursor' }
    ];

    let lineIndex = 0;

    function addLine() {
        if (lineIndex >= lines.length) return;

        const line = lines[lineIndex];
        lineIndex++;

        if (line.type === 'blank') {
            output.appendChild(document.createElement('br'));
            setTimeout(addLine, 100);
            return;
        }

        if (line.type === 'cursor') {
            const span = document.createElement('span');
            span.className = 't-cursor';
            span.textContent = '█';
            output.appendChild(span);
            return;
        }

        const span = document.createElement('span');
        const classMap = {
            prompt: 't-prompt',
            dim: 't-dim',
            success: 't-success'
        };

        // Split prompt lines to colour the arrow/check differently
        if (line.type === 'prompt') {
            const arrow = document.createElement('span');
            arrow.className = 't-prompt';
            arrow.textContent = '→';
            const rest = document.createElement('span');
            rest.className = 't-dim';
            // Highlight the filename
            const parts = line.text.replace('→ ', '').split(' ');
            const verb = parts[0]; // "Running"
            const file = parts.slice(1).join(' ');
            rest.innerHTML = ' <span class="t-dim">' + verb + '</span> <span class="t-file">' + file + '</span>';
            output.appendChild(arrow);
            output.appendChild(rest);
        } else if (line.type === 'success') {
            const check = document.createElement('span');
            check.className = 't-success';
            check.textContent = '✓';
            const rest = document.createElement('span');
            // Parse value segments
            const textAfterCheck = line.text.replace('✓ ', '');
            const colonIdx = textAfterCheck.indexOf(':');
            if (colonIdx > -1) {
                const label = textAfterCheck.substring(0, colonIdx + 1);
                const value = textAfterCheck.substring(colonIdx + 1);
                rest.innerHTML = ' <span class="t-dim">' + label + '</span><span class="t-value">' + value + '</span>';
            } else {
                rest.innerHTML = ' <span class="t-dim">' + textAfterCheck + '</span>';
            }
            output.appendChild(check);
            output.appendChild(rest);
        } else {
            span.className = classMap[line.type] || 't-dim';
            span.textContent = line.text;
            output.appendChild(span);
        }

        output.appendChild(document.createElement('br'));
        const delay = line.type === 'success' ? 600 : line.type === 'dim' ? 400 : 300;
        setTimeout(addLine, delay);
    }

    // Start after a brief delay
    setTimeout(addLine, 800);
});

// ============================================
// Skills Tab Switching (ARIA-compliant)
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    const tabs = document.querySelectorAll('.tab');
    const panels = document.querySelectorAll('.panel');

    function activateTab(tab) {
        // Deactivate all
        tabs.forEach(t => {
            t.classList.remove('active');
            t.setAttribute('aria-selected', 'false');
            t.setAttribute('tabindex', '-1');
        });
        panels.forEach(p => {
            p.classList.remove('active');
            p.hidden = true;
        });

        // Activate selected
        tab.classList.add('active');
        tab.setAttribute('aria-selected', 'true');
        tab.setAttribute('tabindex', '0');

        const panelId = tab.getAttribute('aria-controls');
        const panel = document.getElementById(panelId);
        if (panel) {
            panel.classList.add('active');
            panel.hidden = false;
        }
    }

    tabs.forEach(tab => {
        tab.addEventListener('click', () => activateTab(tab));

        // Keyboard navigation
        tab.addEventListener('keydown', (e) => {
            const tabsArr = Array.from(tabs);
            const idx = tabsArr.indexOf(tab);
            let newIdx = idx;

            if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                e.preventDefault();
                newIdx = (idx + 1) % tabsArr.length;
            } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                e.preventDefault();
                newIdx = (idx - 1 + tabsArr.length) % tabsArr.length;
            } else if (e.key === 'Home') {
                e.preventDefault();
                newIdx = 0;
            } else if (e.key === 'End') {
                e.preventDefault();
                newIdx = tabsArr.length - 1;
            }

            if (newIdx !== idx) {
                activateTab(tabsArr[newIdx]);
                tabsArr[newIdx].focus();
            }
        });
    });

    // Set initial tabindex
    tabs.forEach((tab, i) => {
        tab.setAttribute('tabindex', i === 0 ? '0' : '-1');
    });
});

// ============================================
// Expand/Collapse Repositories
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('expand-repos');
    const list = document.getElementById('all-repos');
    if (!btn || !list) return;

    btn.addEventListener('click', () => {
        const expanded = btn.getAttribute('aria-expanded') === 'true';
        btn.setAttribute('aria-expanded', !expanded);
        list.hidden = expanded;
    });
});

// ============================================
// Intersection Observer — Scroll Reveals
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    const reveals = document.querySelectorAll('[data-reveal]');
    if (!reveals.length) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.08,
        rootMargin: '0px 0px -40px 0px'
    });

    reveals.forEach(el => observer.observe(el));
});

// ============================================
// Contact Form
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('contact-form');
    if (!form) return;

    form.addEventListener('submit', (e) => {
        e.preventDefault();

        const name = form.elements.name.value.trim();
        const email = form.elements.email.value.trim();
        const message = form.elements.message.value.trim();

        if (!name || name.length < 2) {
            showFormMessage('Please enter a valid name (at least 2 characters).', 'error');
            return;
        }
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
            showFormMessage('Please enter a valid email address.', 'error');
            return;
        }
        if (!message || message.length < 10) {
            showFormMessage('Please enter a message (at least 10 characters).', 'error');
            return;
        }

        // Open mail client
        const subject = encodeURIComponent('Portfolio Contact: ' + name);
        const body = encodeURIComponent('Name: ' + name + '\nEmail: ' + email + '\n\nMessage:\n' + message);
        window.location.href = 'mailto:as46g22@soton.ac.uk?subject=' + subject + '&body=' + body;

        setTimeout(() => {
            showFormMessage('Email client opened — please send the email to complete your message.', 'success');
            form.reset();
        }, 1000);
    });
});

function showFormMessage(text, type) {
    const existing = document.querySelector('.form-message');
    if (existing) existing.remove();

    const msg = document.createElement('div');
    msg.className = 'form-message form-message--' + type;
    msg.textContent = text;

    const form = document.getElementById('contact-form');
    if (form) form.parentNode.insertBefore(msg, form);

    setTimeout(() => { if (msg.parentNode) msg.remove(); }, 5000);
}
