document.addEventListener('DOMContentLoaded', function () {
    const demoButton = document.getElementById('demo-button');
    const popup = document.getElementById('popup');
    const closePopup = document.getElementById('close-popup');
    const uploadForm = document.querySelector('.input-file-form');
    const resultsContainer = document.querySelector('.results-container');

    demoButton.addEventListener('click', function () {
        popup.style.display = 'flex';
    });

    closePopup.addEventListener('click', function () {
        popup.style.display = 'none';
    });

    window.addEventListener('click', function (event) {
        if (event.target === popup) {
            popup.style.display = 'none';
        }
    });

    uploadForm.addEventListener('submit', function (event) {
        event.preventDefault();
        const formData = new FormData(uploadForm);

        fetch(uploadForm.action, {
            method: 'POST',
            body: formData
        })
            .then(response => response.text())
            .then(data => {
                resultsContainer.innerHTML = data;
                uploadForm.style.display = 'none';
                setConfidenceBars();
            })
            .catch(error => console.error('Error:', error));
    });

    resultsContainer.addEventListener('click', function (event) {
        if (event.target.classList.contains('class-option')) {
            event.preventDefault();
            const classId = event.target.getAttribute('data-class-id');
            if (classId) {
                fetch(`/details/${classId}/`)
                    .then(response => response.text())
                    .then(data => {
                        resultsContainer.innerHTML = data;
                        setConfidenceBars();
                    })
                    .catch(error => console.error('Error:', error));
            } else if (event.target.id === 'other-option') {
                fetch('/all_classes/')
                    .then(response => response.text())
                    .then(data => {
                        resultsContainer.innerHTML = data;
                        setConfidenceBars();
                    })
                    .catch(error => console.error('Error:', error));
            }
        }
    });

    document.querySelector('.input-file input[type=file]').addEventListener('change', function () {
        const file = this.files[0];
        if (file) {
            document.querySelector('.input-file-text').textContent = file.name;
        }
    });

    function setConfidenceBars() {
        const bars = document.querySelectorAll('.confidence-bar');
        bars.forEach(bar => {
            const confidence = bar.getAttribute('data-confidence');
            const confidencePercent = confidence * 100;
            console.log(`Confidence: ${confidence}, Percent: ${confidencePercent}%`);
            bar.style.width = `${confidencePercent}%`;
        });
    }

    setConfidenceBars();
});
