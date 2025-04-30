document.addEventListener('DOMContentLoaded', function() {
    const askButton = document.getElementById('askButton');
    const questionInput = document.getElementById('question');
    const loadingDiv = document.getElementById('loading');
    const answerDiv = document.getElementById('answer');
    const answerContent = document.getElementById('answerContent');
    const errorDiv = document.getElementById('error');

    askButton.addEventListener('click', async function() {
        const question = questionInput.value.trim();
        
        if (!question) {
            showError('Please enter a question');
            return;
        }

        // Reset UI
        hideError();
        hideAnswer();
        showLoading();

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'An error occurred');
            }

            // Display the answer
            answerContent.textContent = data.answer;
            showAnswer();
        } catch (error) {
            showError(error.message);
        } finally {
            hideLoading();
        }
    });

    function showLoading() {
        loadingDiv.classList.remove('d-none');
    }

    function hideLoading() {
        loadingDiv.classList.add('d-none');
    }

    function showAnswer() {
        answerDiv.classList.remove('d-none');
    }

    function hideAnswer() {
        answerDiv.classList.add('d-none');
    }

    function showError(message) {
        errorDiv.textContent = message;
        errorDiv.classList.remove('d-none');
    }

    function hideError() {
        errorDiv.classList.add('d-none');
    }
}); 