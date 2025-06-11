// training.js
document.addEventListener('DOMContentLoaded', () => {
    const countElement = document.getElementById('count');
    const targetElement = document.getElementById('target');
    const fatigueElement = document.getElementById('fatigue');
    const qualityElement = document.getElementById('quality');
    const feedbackElement = document.getElementById('feedback');
    const targetForm = document.getElementById('targetForm');
    const targetInput = document.getElementById('targetInput');

    // Fetch exercise stats every second
    function updateStats() {
        fetch('/get_count')
            .then(response => {
                if (!response.ok) throw new Error('Network response was not ok');
                return response.json();
            })
            .then(data => {
                countElement.textContent = data.count;
                targetElement.textContent = data.target;
                fatigueElement.textContent = data.fatigue.toFixed(2);
                qualityElement.textContent = data.count;
                feedbackElement.textContent = data.feedback;
            })
            .catch(error => {
                console.error('Error updating stats:', error);
                feedbackElement.textContent = 'Error fetching stats. Please try again.';
            });
    }

    // Start polling
    setInterval(updateStats, 1000);
    // Initial fetch
    updateStats();

    // Handle target submission
    targetForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const target = parseInt(targetInput.value);
        if (isNaN(target) || target <= 0) {
            feedbackElement.textContent = 'Please enter a valid number of reps.';
            return;
        }

        const csrfToken = document.querySelector('input[name="csrf_token"]').value;

        fetch('/set_target', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': csrfToken
            },
            body: JSON.stringify({ target })
        })
            .then(response => {
                if (!response.ok) throw new Error('Failed to set target');
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    feedbackElement.textContent = `Target set to ${data.target}!`;
                    targetInput.value = '';
                } else {
                    feedbackElement.textContent = data.message' || 'Failed to set target.';
                }
            })
            .catch(error => {
                console.error('Error setting target:', error);
                feedbackElement.textContent = 'Error setting target. Please try again.';
            });
    });
});
