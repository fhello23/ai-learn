// ===== NAVIGATION =====
const nav = document.getElementById('nav');
const navToggle = document.getElementById('navToggle');
const navLinks = document.getElementById('navLinks');

// Scroll effect
let lastScroll = 0;
window.addEventListener('scroll', () => {
    const scrollY = window.scrollY;
    nav.classList.toggle('nav--scrolled', scrollY > 50);
    lastScroll = scrollY;
});

// Mobile toggle
navToggle.addEventListener('click', () => {
    navLinks.classList.toggle('nav__links--open');
});

// Close mobile nav on link click
navLinks.querySelectorAll('.nav__link').forEach(link => {
    link.addEventListener('click', () => {
        navLinks.classList.remove('nav__links--open');
    });
});

// Active link tracking
const sections = document.querySelectorAll('.section, .hero');
const navLinkElements = document.querySelectorAll('.nav__link');

const observerOptions = {
    root: null,
    rootMargin: '-20% 0px -60% 0px',
    threshold: 0
};

const sectionObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const id = entry.target.id;
            navLinkElements.forEach(link => {
                link.classList.toggle('nav__link--active',
                    link.getAttribute('href') === `#${id}`
                );
            });
        }
    });
}, observerOptions);

sections.forEach(section => {
    if (section.id) sectionObserver.observe(section);
});

// ===== SCROLL ANIMATIONS =====
const animateElements = document.querySelectorAll('[data-animate]');

const animateObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const delay = entry.target.dataset.delay || 0;
            setTimeout(() => {
                entry.target.classList.add('animate-in');
            }, parseInt(delay));
            animateObserver.unobserve(entry.target);
        }
    });
}, {
    root: null,
    rootMargin: '0px 0px -80px 0px',
    threshold: 0.1
});

animateElements.forEach(el => animateObserver.observe(el));

// ===== COUNTER ANIMATION =====
const counters = document.querySelectorAll('[data-count]');

const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const target = parseInt(entry.target.dataset.count);
            animateCounter(entry.target, target);
            counterObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.5 });

counters.forEach(counter => counterObserver.observe(counter));

function animateCounter(element, target) {
    const duration = 1500;
    const start = performance.now();

    function update(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        element.textContent = Math.round(target * eased);

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

// ===== TABS =====
const tabButtons = document.querySelectorAll('.tabs__btn');
const tabPanels = document.querySelectorAll('.tabs__panel');

tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const tabId = btn.dataset.tab;

        tabButtons.forEach(b => b.classList.remove('tabs__btn--active'));
        tabPanels.forEach(p => p.classList.remove('tabs__panel--active'));

        btn.classList.add('tabs__btn--active');
        document.getElementById(`tab-${tabId}`).classList.add('tabs__panel--active');
    });
});

// ===== ALGORITHM FILTER =====
const filterButtons = document.querySelectorAll('.algo-filter__btn');
const whenCards = document.querySelectorAll('.when-card');

filterButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const filter = btn.dataset.filter;

        filterButtons.forEach(b => b.classList.remove('algo-filter__btn--active'));
        btn.classList.add('algo-filter__btn--active');

        whenCards.forEach(card => {
            if (filter === 'all' || card.dataset.category === filter) {
                card.classList.remove('hidden');
            } else {
                card.classList.add('hidden');
            }
        });
    });
});

// ===== NEURAL NETWORK CANVAS =====
const canvas = document.getElementById('neuralCanvas');
if (canvas) {
    const ctx = canvas.getContext('2d');
    let width, height;
    let particles = [];

    let animationFrame;

    function resize() {
        width = canvas.width = canvas.offsetWidth;
        height = canvas.height = canvas.offsetHeight;
    }

    function initParticles() {
        particles = [];
        const count = Math.min(Math.floor((width * height) / 15000), 80);

        for (let i = 0; i < count; i++) {
            particles.push({
                x: Math.random() * width,
                y: Math.random() * height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                radius: Math.random() * 2 + 1,
                opacity: Math.random() * 0.5 + 0.2
            });
        }
    }

    function drawParticles() {
        ctx.clearRect(0, 0, width, height);

        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 150) {
                    const opacity = (1 - dist / 150) * 0.15;
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(129, 140, 248, ${opacity})`;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }
        }

        // Draw particles
        particles.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(129, 140, 248, ${p.opacity})`;
            ctx.fill();

            // Glow
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius * 3, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(129, 140, 248, ${p.opacity * 0.1})`;
            ctx.fill();
        });
    }

    function updateParticles() {
        particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0 || p.x > width) p.vx *= -1;
            if (p.y < 0 || p.y > height) p.vy *= -1;

            p.x = Math.max(0, Math.min(width, p.x));
            p.y = Math.max(0, Math.min(height, p.y));
        });
    }

    function animate() {
        updateParticles();
        drawParticles();
        animationFrame = requestAnimationFrame(animate);
    }

    // Only run canvas animation when hero is visible
    const heroObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                if (!animationFrame) animate();
            } else {
                if (animationFrame) {
                    cancelAnimationFrame(animationFrame);
                    animationFrame = null;
                }
            }
        });
    }, { threshold: 0 });

    resize();
    initParticles();
    heroObserver.observe(canvas.closest('.hero'));

    window.addEventListener('resize', () => {
        resize();
        initParticles();
    });
}

// ===== SMOOTH SCROLL =====
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

// ===== NEURAL NETWORK DIAGRAM CONNECTIONS =====
function drawNNConnections() {
    const diagram = document.querySelector('.nn-visual__diagram');
    if (!diagram) return;

    const layers = diagram.querySelectorAll('.nn-layer');
    const svgNS = 'http://www.w3.org/2000/svg';

    // Remove existing SVGs
    diagram.querySelectorAll('.nn-svg').forEach(s => s.remove());

    for (let l = 0; l < layers.length - 1; l++) {
        const currentNeurons = layers[l].querySelectorAll('.nn-neuron');
        const nextNeurons = layers[l + 1].querySelectorAll('.nn-neuron');

        const svg = document.createElementNS(svgNS, 'svg');
        svg.classList.add('nn-svg');
        svg.style.position = 'absolute';
        svg.style.top = '0';
        svg.style.left = '0';
        svg.style.width = '100%';
        svg.style.height = '100%';
        svg.style.pointerEvents = 'none';
        svg.style.zIndex = '1';

        const diagramRect = diagram.getBoundingClientRect();

        currentNeurons.forEach(n1 => {
            nextNeurons.forEach(n2 => {
                const r1 = n1.getBoundingClientRect();
                const r2 = n2.getBoundingClientRect();

                const x1 = r1.left + r1.width / 2 - diagramRect.left;
                const y1 = r1.top + r1.height / 2 - diagramRect.top;
                const x2 = r2.left + r2.width / 2 - diagramRect.left;
                const y2 = r2.top + r2.height / 2 - diagramRect.top;

                const line = document.createElementNS(svgNS, 'line');
                line.setAttribute('x1', x1);
                line.setAttribute('y1', y1);
                line.setAttribute('x2', x2);
                line.setAttribute('y2', y2);
                line.setAttribute('stroke', 'rgba(129, 140, 248, 0.12)');
                line.setAttribute('stroke-width', '1');
                svg.appendChild(line);
            });
        });

        diagram.appendChild(svg);
    }
}

// Draw connections after layout settles
window.addEventListener('load', () => {
    setTimeout(drawNNConnections, 300);
});
window.addEventListener('resize', () => {
    setTimeout(drawNNConnections, 100);
});

// ===== NEURON HOVER EFFECT =====
document.querySelectorAll('.nn-neuron').forEach(neuron => {
    neuron.addEventListener('mouseenter', () => {
        neuron.style.transform = 'scale(1.4)';
        neuron.style.boxShadow = '0 0 24px rgba(99, 102, 241, 0.6)';
    });
    neuron.addEventListener('mouseleave', () => {
        neuron.style.transform = 'scale(1)';
        neuron.style.boxShadow = '';
    });
});

// ===== CARD TILT EFFECT =====
document.querySelectorAll('.card--glass, .arch-card').forEach(card => {
    card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;

        const rotateX = (y - centerY) / centerY * -3;
        const rotateY = (x - centerX) / centerX * 3;

        card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-4px)`;
    });

    card.addEventListener('mouseleave', () => {
        card.style.transform = '';
    });
});

// ===== ALGORITHM DETAIL NAVIGATION =====
document.querySelectorAll('[data-algo]').forEach(card => {
    card.addEventListener('click', () => {
        window.location.href = `algorithm.html?id=${card.dataset.algo}`;
    });
    card.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            window.location.href = `algorithm.html?id=${card.dataset.algo}`;
        }
    });
});

// ===== KEYBOARD NAVIGATION =====
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        navLinks.classList.remove('nav__links--open');
    }
});
