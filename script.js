// Mobile Navigation Toggle
const hamburger = document.getElementById('hamburger');
const navMenu = document.getElementById('nav-menu');

hamburger.addEventListener('click', () => {
    hamburger.classList.toggle('active');
    navMenu.classList.toggle('active');
});

// Close mobile menu when clicking on a link
document.querySelectorAll('.nav-link').forEach(n => n.addEventListener('click', () => {
    hamburger.classList.remove('active');
    navMenu.classList.remove('active');
}));

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Typing Animation
const typedTextSpan = document.querySelector(".typed-text");
const cursorSpan = document.querySelector(".cursor");

const textArray = ["Computational Physicist", "Data Analyst", "Research Scientist", "Quant Developer", "Machine Learning Engineer"];
const typingDelay = 200;
const erasingDelay = 100;
const newTextDelay = 2000;
let textArrayIndex = 0;
let charIndex = 0;

function type() {
    if (charIndex < textArray[textArrayIndex].length) {
        if (!cursorSpan.classList.contains("typing")) cursorSpan.classList.add("typing");
        typedTextSpan.textContent += textArray[textArrayIndex].charAt(charIndex);
        charIndex++;
        setTimeout(type, typingDelay);
    } else {
        cursorSpan.classList.remove("typing");
        setTimeout(erase, newTextDelay);
    }
}

function erase() {
    if (charIndex > 0) {
        if (!cursorSpan.classList.contains("typing")) cursorSpan.classList.add("typing");
        typedTextSpan.textContent = textArray[textArrayIndex].substring(0, charIndex - 1);
        charIndex--;
        setTimeout(erase, erasingDelay);
    } else {
        cursorSpan.classList.remove("typing");
        textArrayIndex++;
        if (textArrayIndex >= textArray.length) textArrayIndex = 0;
        setTimeout(type, typingDelay + 1100);
    }
}

document.addEventListener("DOMContentLoaded", function() {
    if (textArray.length) setTimeout(type, newTextDelay + 250);
});

// Navbar background on scroll
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.style.background = 'rgba(15, 20, 25, 0.98)';
        navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.3)';
    } else {
        navbar.style.background = 'rgba(15, 20, 25, 0.95)';
        navbar.style.boxShadow = 'none';
    }
});

// Intersection Observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Animate elements on scroll
document.addEventListener('DOMContentLoaded', () => {
    const animateElements = document.querySelectorAll('.skill-category, .project-card, .stat');
    
    animateElements.forEach(element => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(30px)';
        element.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(element);
    });
});

// Contact form handling
const contactForm = document.querySelector('.contact-form');
contactForm.addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Get form data
    const formData = new FormData(contactForm);
    const name = formData.get('name');
    const email = formData.get('email');
    const message = formData.get('message');
    
    // Simple validation
    if (!name || !email || !message) {
        alert('Please fill in all fields.');
        return;
    }
    
    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
        alert('Please enter a valid email address.');
        return;
    }
    
    // Simulate form submission
    const submitButton = contactForm.querySelector('button[type="submit"]');
    const originalText = submitButton.textContent;
    submitButton.textContent = 'Sending...';
    submitButton.disabled = true;
    
    setTimeout(() => {
        alert('Thank you for your message! I\'ll get back to you soon.');
        contactForm.reset();
        submitButton.textContent = originalText;
        submitButton.disabled = false;
    }, 1500);
});

// Contact Form Functionality
document.addEventListener('DOMContentLoaded', () => {
    const contactForm = document.querySelector('.contact-form');
    
    if (contactForm) {
        contactForm.addEventListener('submit', handleFormSubmit);
    }
});

function handleFormSubmit(e) {
    e.preventDefault();
    
    const formData = {
        name: document.getElementById('name').value.trim(),
        email: document.getElementById('email').value.trim(),
        message: document.getElementById('message').value.trim()
    };
    
    // Validate form data
    if (!validateForm(formData)) {
        return;
    }
    
    // Show loading state
    showLoadingState();
    
    // Choose your preferred submission method:
    // Option 1: Email client (most reliable, no backend needed)
    submitViaEmailClient(formData);
    
    // Option 2: Formspree (uncomment to use)
    // submitViaFormspree(formData);
    
    // Option 3: EmailJS (uncomment to use)
    // submitViaEmailJS(formData);
}

function validateForm(data) {
    const { name, email, message } = data;
    
    if (!name || name.length < 2) {
        showMessage('Please enter a valid name (at least 2 characters)', 'error');
        return false;
    }
    
    if (!isValidEmail(email)) {
        showMessage('Please enter a valid email address', 'error');
        return false;
    }
    
    if (!message || message.length < 10) {
        showMessage('Please enter a message (at least 10 characters)', 'error');
        return false;
    }
    
    return true;
}

function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

// Option 1: Email Client (Most Reliable)
function submitViaEmailClient(data) {
    const subject = `Portfolio Contact: ${data.name}`;
    const body = `Name: ${data.name}\nEmail: ${data.email}\n\nMessage:\n${data.message}`;
    const mailtoLink = `mailto:as46g22@soton.ac.uk?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
    
    window.location.href = mailtoLink;
    
    setTimeout(() => {
        showMessage('Email client opened! Please send the email to complete your message.', 'success');
        resetForm();
        hideLoadingState();
    }, 1000);
}

// Option 2: Formspree (Free service, requires setup)
async function submitViaFormspree(data) {
    try {
        const response = await fetch('https://formspree.io/f/YOUR_FORM_ID', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (response.ok) {
            showMessage('Message sent successfully! I\'ll get back to you soon.', 'success');
            resetForm();
        } else {
            throw new Error('Failed to send message');
        }
    } catch (error) {
        showMessage('Failed to send message. Please try again or contact me directly.', 'error');
    } finally {
        hideLoadingState();
    }
}

// Option 3: EmailJS (Free service, requires setup)
async function submitViaEmailJS(data) {
    try {
        // You need to include EmailJS library and configure it
        await emailjs.send('YOUR_SERVICE_ID', 'YOUR_TEMPLATE_ID', {
            from_name: data.name,
            from_email: data.email,
            message: data.message,
            to_email: 'as46g22@soton.ac.uk'
        });
        
        showMessage('Message sent successfully! I\'ll get back to you soon.', 'success');
        resetForm();
    } catch (error) {
        showMessage('Failed to send message. Please try again or contact me directly.', 'error');
    } finally {
        hideLoadingState();
    }
}

function showLoadingState() {
    const submitBtn = document.querySelector('.contact-form button[type="submit"]');
    if (submitBtn) {
        submitBtn.textContent = 'Sending...';
        submitBtn.disabled = true;
        submitBtn.style.opacity = '0.7';
    }
}

function hideLoadingState() {
    const submitBtn = document.querySelector('.contact-form button[type="submit"]');
    if (submitBtn) {
        submitBtn.textContent = 'Send Message';
        submitBtn.disabled = false;
        submitBtn.style.opacity = '1';
    }
}

function showMessage(text, type) {
    // Remove any existing messages
    const existingMessage = document.querySelector('.form-message');
    if (existingMessage) {
        existingMessage.remove();
    }
    
    // Create new message element
    const message = document.createElement('div');
    message.className = `form-message form-message--${type}`;
    message.textContent = text;
    
    // Insert message before the form
    const contactForm = document.querySelector('.contact-form');
    contactForm.parentNode.insertBefore(message, contactForm);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (message.parentNode) {
            message.remove();
        }
    }, 5000);
}

function resetForm() {
    const contactForm = document.querySelector('.contact-form');
    if (contactForm) {
        contactForm.reset();
    }
}

// Add active class to current nav item based on scroll position
window.addEventListener('scroll', () => {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link');
    
    let current = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop - 100;
        const sectionHeight = section.clientHeight;
        
        if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
            current = section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// Parallax effect for hero section
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const hero = document.querySelector('.hero');
    const rate = scrolled * -0.5;
    
    if (hero) {
        hero.style.transform = `translateY(${rate}px)`;
    }
});

// Add smooth reveal animation to sections
const revealSections = document.querySelectorAll('section');
const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('revealed');
        }
    });
}, {
    threshold: 0.15
});

revealSections.forEach(section => {
    section.classList.add('hidden');
    revealObserver.observe(section);
});

// Add CSS for reveal animation
const style = document.createElement('style');
style.textContent = `
    .hidden {
        opacity: 0;
        transform: translateY(50px);
        transition: all 0.8s ease;
    }
    
    .revealed {
        opacity: 1;
        transform: translateY(0);
    }
    
    .nav-link.active {
        color: #2563eb;
    }
    
    .nav-link.active::after {
        width: 100%;
    }
`;
document.head.appendChild(style);

// Particle effect for physics theme
function createParticles() {
    const particlesContainer = document.createElement('div');
    particlesContainer.className = 'particles';
    particlesContainer.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    `;
    
    document.body.appendChild(particlesContainer);
    
    for (let i = 0; i < 50; i++) {
        const particle = document.createElement('div');
        particle.style.cssText = `
            position: absolute;
            width: 2px;
            height: 2px;
            background: rgba(96, 165, 250, 0.3);
            border-radius: 50%;
            animation: float-particle ${Math.random() * 20 + 10}s linear infinite;
            top: ${Math.random() * 100}vh;
            left: ${Math.random() * 100}vw;
            animation-delay: ${Math.random() * 20}s;
        `;
        particlesContainer.appendChild(particle);
    }
}

// Add particle animation CSS
const particleStyle = document.createElement('style');
particleStyle.textContent = `
    @keyframes float-particle {
        0% {
            transform: translateY(0) translateX(0);
            opacity: 0;
        }
        10% {
            opacity: 0.3;
        }
        90% {
            opacity: 0.1;
        }
        100% {
            transform: translateY(-100vh) translateX(${Math.random() * 200 - 100}px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(particleStyle);

// Initialize particles when page loads
document.addEventListener('DOMContentLoaded', createParticles);

// Sierpiński Triangle Fractal Animation
class SierpinskiTriangle {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.points = [];
        this.currentPoint = null;
        this.vertices = [];
        this.animationFrame = 0;
        this.init();
    }

    init() {
        this.resizeCanvas();
        this.setupVertices();
        this.currentPoint = {
            x: this.canvas.width / 2,
            y: this.canvas.height / 2
        };
        this.animate();
        
        // Handle window resize
        window.addEventListener('resize', () => {
            this.resizeCanvas();
            this.setupVertices();
            this.points = [];
        });
    }

    resizeCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    setupVertices() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        const size = Math.min(width, height) * 0.8;
        const centerX = width / 2;
        const centerY = height / 2;

        // Three vertices of an equilateral triangle
        this.vertices = [
            { x: centerX, y: centerY - size / 2 }, // Top
            { x: centerX - size / 2, y: centerY + size / 2 }, // Bottom left
            { x: centerX + size / 2, y: centerY + size / 2 }  // Bottom right
        ];
    }

    drawPoint(x, y, alpha = 1) {
        this.ctx.fillStyle = `rgba(96, 165, 250, ${alpha})`;
        this.ctx.fillRect(x, y, 1, 1);
    }

    animate() {
        // Clear canvas with fade effect
        this.ctx.fillStyle = 'rgba(12, 16, 32, 0.02)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Generate multiple points per frame for faster generation
        for (let i = 0; i < 50; i++) {
            // Choose random vertex
            const randomVertex = this.vertices[Math.floor(Math.random() * 3)];
            
            // Move halfway to the chosen vertex
            this.currentPoint.x = (this.currentPoint.x + randomVertex.x) / 2;
            this.currentPoint.y = (this.currentPoint.y + randomVertex.y) / 2;
            
            // Draw the point
            this.drawPoint(this.currentPoint.x, this.currentPoint.y, 0.8);
            
            // Store point for potential effects
            this.points.push({
                x: this.currentPoint.x,
                y: this.currentPoint.y,
                age: 0
            });
        }

        // Limit points array size for performance
        if (this.points.length > 10000) {
            this.points = this.points.slice(-5000);
        }

        // Add subtle glow effect to recent points
        this.points.forEach((point, index) => {
            if (point.age < 100) {
                const alpha = (100 - point.age) / 100 * 0.3;
                this.drawPoint(point.x, point.y, alpha);
                point.age++;
            }
        });

        this.animationFrame++;
        requestAnimationFrame(() => this.animate());
    }
}

// Initialize Sierpiński Triangle when page loads
document.addEventListener('DOMContentLoaded', () => {
    new SierpinskiTriangle('sierpinski-canvas');
});

// Mobile Projects Section Fix
function ensureMobileProjectsVisibility() {
    // Check if we're on mobile
    if (window.innerWidth <= 768) {
        const projectsSection = document.querySelector('.projects');
        const projectsGrid = document.querySelector('.projects-grid');
        const projectCards = document.querySelectorAll('.project-card');
        
        if (projectsSection) {
            // Ultra-aggressive projects section styling
            projectsSection.style.cssText = `
                background: #1a1f2e !important;
                background-color: #1a1f2e !important;
                background-image: none !important;
                backdrop-filter: none !important;
                -webkit-backdrop-filter: none !important;
                padding: 60px 0 !important;
                display: block !important;
                visibility: visible !important;
                opacity: 1 !important;
                min-height: auto !important;
                position: relative !important;
                z-index: auto !important;
                overflow: visible !important;
            `;
        }
        
        if (projectsGrid) {
            projectsGrid.style.cssText = `
                display: block !important;
                width: 100% !important;
                margin: 0 !important;
                padding: 0 15px !important;
                box-sizing: border-box !important;
            `;
        }
        
        // Completely rebuild each project card
        projectCards.forEach((card, index) => {
            // Remove any existing mobile background fixes
            const existingBgs = card.querySelectorAll('.mobile-bg-fix, .mobile-img-bg, .mobile-content-bg');
            existingBgs.forEach(bg => bg.remove());
            
            // Reset all styles and rebuild from scratch
            card.style.cssText = `
                all: unset !important;
                display: block !important;
                width: 100% !important;
                margin: 0 0 20px 0 !important;
                border: 3px solid #60a5fa !important;
                border-radius: 8px !important;
                overflow: hidden !important;
                box-sizing: border-box !important;
                background: #1e3a5f !important;
                background-color: #1e3a5f !important;
                background-image: none !important;
                opacity: 1 !important;
                visibility: visible !important;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5) !important;
                position: relative !important;
                z-index: ${index + 1} !important;
                transform: none !important;
                transition: none !important;
                -webkit-transform: translateZ(0) !important;
                transform: translateZ(0) !important;
                min-height: 300px !important;
                font-family: inherit !important;
            `;
            
            // Create and insert primary background layer
            const primaryBg = document.createElement('div');
            primaryBg.className = 'mobile-primary-bg';
            primaryBg.style.cssText = `
                position: absolute !important;
                top: 0 !important;
                left: 0 !important;
                width: 100% !important;
                height: 100% !important;
                background: #1e3a5f !important;
                background-color: #1e3a5f !important;
                background-image: none !important;
                z-index: -10 !important;
                pointer-events: none !important;
                opacity: 1 !important;
            `;
            card.insertBefore(primaryBg, card.firstChild);
            
            // Handle project image
            const projectImage = card.querySelector('.project-image');
            if (projectImage) {
                projectImage.style.cssText = `
                    all: unset !important;
                    display: block !important;
                    width: 100% !important;
                    height: 120px !important;
                    background: #2563eb !important;
                    background-color: #2563eb !important;
                    background-image: none !important;
                    border-radius: 5px 5px 0 0 !important;
                    position: relative !important;
                    overflow: hidden !important;
                    box-sizing: border-box !important;
                    margin: 0 !important;
                    padding: 0 !important;
                `;
                
                // Add image background layer
                const imgBg = document.createElement('div');
                imgBg.className = 'mobile-img-bg';
                imgBg.style.cssText = `
                    position: absolute !important;
                    top: 0 !important;
                    left: 0 !important;
                    width: 100% !important;
                    height: 100% !important;
                    background: #2563eb !important;
                    background-color: #2563eb !important;
                    z-index: -1 !important;
                `;
                projectImage.insertBefore(imgBg, projectImage.firstChild);
            }
            
            // Handle project content
            const projectContent = card.querySelector('.project-content');
            if (projectContent) {
                projectContent.style.cssText = `
                    all: unset !important;
                    display: block !important;
                    width: 100% !important;
                    padding: 15px !important;
                    background: #1e3a5f !important;
                    background-color: #1e3a5f !important;
                    background-image: none !important;
                    border-radius: 0 0 5px 5px !important;
                    box-sizing: border-box !important;
                    min-height: 180px !important;
                    position: relative !important;
                    overflow: hidden !important;
                    font-family: inherit !important;
                    color: white !important;
                `;
                
                // Add content background layer
                const contentBg = document.createElement('div');
                contentBg.className = 'mobile-content-bg';
                contentBg.style.cssText = `
                    position: absolute !important;
                    top: 0 !important;
                    left: 0 !important;
                    width: 100% !important;
                    height: 100% !important;
                    background: #1e3a5f !important;
                    background-color: #1e3a5f !important;
                    z-index: -1 !important;
                `;
                projectContent.insertBefore(contentBg, projectContent.firstChild);
                
                // Ensure text elements are visible
                const title = projectContent.querySelector('h3');
                const description = projectContent.querySelector('p');
                const techTags = projectContent.querySelectorAll('.tech-tag');
                const projectLinks = projectContent.querySelectorAll('.project-link');
                
                if (title) {
                    title.style.cssText = `
                        color: #ffffff !important;
                        font-size: 1.1rem !important;
                        margin: 0 0 8px 0 !important;
                        padding: 0 !important;
                        font-weight: 600 !important;
                        line-height: 1.3 !important;
                        display: block !important;
                        position: relative !important;
                        z-index: 1 !important;
                        font-family: inherit !important;
                    `;
                }
                
                if (description) {
                    description.style.cssText = `
                        color: #e5e7eb !important;
                        font-size: 0.9rem !important;
                        line-height: 1.4 !important;
                        margin: 0 0 12px 0 !important;
                        padding: 0 !important;
                        display: block !important;
                        position: relative !important;
                        z-index: 1 !important;
                        font-family: inherit !important;
                    `;
                }
                
                // Style tech tags
                techTags.forEach(tag => {
                    tag.style.cssText = `
                        display: inline-block !important;
                        background: #60a5fa !important;
                        background-color: #60a5fa !important;
                        color: #1e3a5f !important;
                        padding: 3px 8px !important;
                        border-radius: 10px !important;
                        font-size: 0.75rem !important;
                        font-weight: 600 !important;
                        margin: 0 5px 5px 0 !important;
                        font-family: inherit !important;
                    `;
                });
                
                // Style project links
                projectLinks.forEach(link => {
                    link.style.cssText = `
                        display: inline-flex !important;
                        align-items: center !important;
                        gap: 4px !important;
                        color: #93c5fd !important;
                        text-decoration: none !important;
                        font-weight: 500 !important;
                        font-size: 0.85rem !important;
                        margin-right: 8px !important;
                        position: relative !important;
                        z-index: 1 !important;
                        font-family: inherit !important;
                    `;
                });
            }
        });
        
        // Force multiple repaints
        setTimeout(() => {
            if (projectsSection) {
                projectsSection.style.transform = 'translateZ(0)';
                projectsSection.offsetHeight; // Force reflow
            }
            projectCards.forEach((card, index) => {
                card.style.transform = 'translateZ(0)';
                card.offsetHeight; // Force reflow
            });
        }, 100);
        
        // Additional repaint after longer delay
        setTimeout(() => {
            projectCards.forEach(card => {
                card.style.willChange = 'auto';
                card.offsetHeight; // Force reflow
            });
        }, 500);
    }
}

// Run on page load and resize
document.addEventListener('DOMContentLoaded', ensureMobileProjectsVisibility);
window.addEventListener('resize', ensureMobileProjectsVisibility);
window.addEventListener('orientationchange', () => {
    setTimeout(ensureMobileProjectsVisibility, 500);
});

// Force run multiple times with increasing delays
setTimeout(ensureMobileProjectsVisibility, 100);
setTimeout(ensureMobileProjectsVisibility, 500);
setTimeout(ensureMobileProjectsVisibility, 1000);
setTimeout(ensureMobileProjectsVisibility, 2000);
setTimeout(ensureMobileProjectsVisibility, 3000);
