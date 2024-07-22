document.addEventListener('DOMContentLoaded', function() {
    const demoLink = document.querySelector('.demo-link');

    // Smooth scroll for demo link
    demoLink.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        target.scrollIntoView({ behavior: 'smooth' });
    });
});
