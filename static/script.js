const messagesEl = document.getElementById('messages');
const formEl = document.getElementById('chatForm');
const inputEl = document.getElementById('messageInput');

function addMessage(text, who) {
    const wrap = document.createElement('div');
    wrap.className = `message ${who}`;
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

formEl.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = (inputEl.value || '').trim();
    if (!text) return;
    addMessage(text, 'user');
    inputEl.value = '';

    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
    })
        .then((res) => res.json())
        .then((data) => {
            if (data.error) {
                addMessage(data.error, 'bot');
            } else {
                addMessage(data.response || '', 'bot');
            }
        })
        .catch(() => {
            addMessage('Something went wrong talking to the server.', 'bot');
        });
});

