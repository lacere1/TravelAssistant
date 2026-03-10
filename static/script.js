const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const intentDisplay = document.getElementById('intentDisplay');
const confidenceDisplay = document.getElementById('confidenceDisplay');
const entitiesDisplay = document.getElementById('entitiesDisplay');

function formatMessage(text) {
    if (!text) return text;

    text = text.replace(/\n/g, '<br>');

    const bulletPattern = /^\s*[-•]\s+(.+)$/gm;
    text = text.replace(bulletPattern, '<li>$1</li>');

    if (text.includes('<li>')) {
        text = text.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    }

    return text;
}

function appendDisruptionCard(disruption) {
    const cardEl = document.createElement('div');
    cardEl.className = 'disruption-card';
    const line = disruption.line || disruption.route || 'Unknown';
    cardEl.innerHTML = `
        <div class="disruption-line">${line}</div>
        <div class="disruption-status">${disruption.status}</div>
        ${disruption.reason ? `<div class="disruption-reason">${disruption.reason}</div>` : ''}
    `;
    chatMessages.appendChild(cardEl);
}

function appendTimetableCard(timetable) {
    const cardEl = document.createElement('div');
    cardEl.className = 'timetable-card';
    let html = '<div class="timetable-header">Timetable</div><div class="timetable-items">';

    if (Array.isArray(timetable)) {
        timetable.forEach(item => {
            const timeStr = item.expected_arrival ? new Date(item.expected_arrival).toLocaleTimeString() : `${item.time_to_station}s`;
            html += `
                <div class="timetable-item">
                    <div class="timetable-line">${item.line}</div>
                    <div class="timetable-dest">${item.destination}</div>
                    <div class="timetable-time">${timeStr}</div>
                </div>
            `;
        });
    }

    html += '</div>';
    cardEl.innerHTML = html;
    chatMessages.appendChild(cardEl);
}

function showTypingIndicator() {
    const typingEl = document.createElement('div');
    typingEl.id = 'typing-indicator';
    typingEl.className = 'message bot-message typing-indicator';
    typingEl.innerHTML = '<span></span><span></span><span></span>';
    chatMessages.appendChild(typingEl);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const typingEl = document.getElementById('typing-indicator');
    if (typingEl) {
        typingEl.remove();
    }
}

function updateInfoPanel(intent, confidence, entities) {
    intentDisplay.textContent = intent || '-';
    confidenceDisplay.textContent = confidence ? (confidence * 100).toFixed(0) + '%' : '-';
    entitiesDisplay.textContent = Object.keys(entities).length > 0
        ? Object.entries(entities).map(([k, v]) => `${k}: ${v}`).join(', ')
        : '-';
}

function addMessage(text, isUser) {
    const messageEl = document.createElement('div');
    messageEl.className = isUser ? 'message user-message' : 'message bot-message';
    messageEl.innerHTML = formatMessage(text);
    chatMessages.appendChild(messageEl);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    addMessage(message, true);
    userInput.value = '';

    showTypingIndicator();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        });

        removeTypingIndicator();
        const data = await response.json();

        if (response.ok) {
            addMessage(data.response || data.message, false);
            updateInfoPanel(data.intent, data.confidence, data.entities || {});

            if (data.disruption && Array.isArray(data.disruption)) {
                data.disruption.forEach(d => appendDisruptionCard(d));
            }

            if (data.timetable && Array.isArray(data.timetable)) {
                appendTimetableCard(data.timetable);
            }
        } else {
            addMessage('Error: ' + (data.error || 'Unknown error'), false);
        }
    } catch (err) {
        removeTypingIndicator();
        addMessage('Connection error', false);
    }
}

sendButton.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendMessage();
    }
});
