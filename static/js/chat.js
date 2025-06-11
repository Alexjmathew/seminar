function sendMessage() {
    const receiver = document.getElementById('receiver').value;
    const message = document.getElementById('message').value;
    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `receiver=${receiver}&message=${message}`
    }).then(response => response.json()).then(data => {
        const output = document.getElementById('chat-output');
        output.innerHTML += `<p><b>You:</b> ${message}</p>`;
        output.innerHTML += `<p><b>RAG Chatbot:</b> ${data.response}</p>`;
        document.getElementById('message').value = '';
    });
}
