document.addEventListener('DOMContentLoaded', function() {
    const demoButton = document.getElementById('demo-button');
    const popup = document.getElementById('popup');
    const closePopup = document.getElementById('close-popup');
    const uploadForm = document.querySelector('.input-file-form');
    const formWrapper = document.querySelector('.form-wrapper');
    const resultsContainer = document.querySelector('.results-container');

    demoButton.addEventListener('click', function() {
        popup.style.display = 'flex';
    });

    closePopup.addEventListener('click', function() {
        popup.style.display = 'none';
    });

    window.addEventListener('click', function(event) {
        if (event.target === popup) {
            popup.style.display = 'none';
        }
    });

    uploadForm.addEventListener('submit', function(event) {
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
        })
        .catch(error => console.error('Error:', error));
    });

    resultsContainer.addEventListener('click', function(event) {
        if (event.target.classList.contains('class-option')) {
            event.preventDefault();
            const classId = event.target.getAttribute('data-class-id');

            fetch(`/details/${classId}/`)
            .then(response => response.text())
            .then(data => {
                resultsContainer.innerHTML = data;
            })
            .catch(error => console.error('Error:', error));
        }
    });

    // Обновление текстового поля с именем выбранного файла
    document.querySelector('.input-file input[type=file]').addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            document.querySelector('.input-file-text').textContent = file.name;
        }
    });
});
