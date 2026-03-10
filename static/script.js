const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const intentDisplay = document.getElementById('intentDisplay');
const confidenceDisplay = document.getElementById('confidenceDisplay');
const entitiesDisplay = document.getElementById('entitiesDisplay');

const journeyInputsSection = document.getElementById('journey-inputs');
const journeyPlannerToggle = document.getElementById('journey-planner-toggle');
const fromInput = document.getElementById('from-input');
const toInput = document.getElementById('to-input');
const planBtn = document.getElementById('plan-btn');

function clearMessages() {
    chatMessages.innerHTML = '';
}

function appendMessage(sender, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.innerHTML = `<div class="message-content"><p>${escapeHtml(text)}</p></div>`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

function displayMetadata(intent, confidence, entities) {
    intentDisplay.textContent = intent || '-';
    confidenceDisplay.textContent = confidence ? (confidence * 100).toFixed(1) + '%' : '-';
    entitiesDisplay.textContent = Object.keys(entities).length > 0 ? JSON.stringify(entities) : '-';
}

function buildJourneyCard(journey) {
    const legs = (journey.legs || [])
        .map(leg => `<span class="journey-leg">${escapeHtml(leg.mode)}</span>`)
        .join('');

    return `
        <div class="journey-card">
            <div class="journey-time">
                <span class="departure">${journey.departure}</span>
                <span class="arrow">→</span>
                <span class="arrival">${journey.arrival}</span>
            </div>
            <div class="journey-duration">${journey.duration} mins</div>
            <div class="journey-legs">${legs}</div>
        </div>
    `;
}

function appendJourneyCards(journeys) {
    if (!journeys || journeys.length === 0) return;

    const container = document.createElement('div');
    container.className = 'journeys-container';
    container.innerHTML = journeys.map(j => buildJourneyCard(j)).join('');
    chatMessages.appendChild(container);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function sendPlannedJourney() {
    const from = fromInput.value.trim();
    const to = toInput.value.trim();

    if (!from || !to) {
        alert('Please fill in both From and To fields.');
        return;
    }

    const payload = {
        message: '',
        from: from,
        to: to
    };

    sendChatMessage(payload);
}

function toggleJourneyPlanner() {
    const isHidden = journeyInputsSection.classList.contains('hidden');
    if (isHidden) {
        journeyInputsSection.classList.remove('hidden');
        journeyPlannerToggle.setAttribute('aria-expanded', 'true');
        journeyPlannerToggle.textContent = 'Hide journey planner';
    } else {
        journeyInputsSection.classList.add('hidden');
        journeyPlannerToggle.setAttribute('aria-expanded', 'false');
        journeyPlannerToggle.textContent = 'Show journey planner';
    }
}

function sendChatMessage(payload) {
    const message = payload.message || '';
    const from = payload.from || '';
    const to = payload.to || '';

    if (!message && !from && !to) return;

    if (message) {
        appendMessage('user', message);
    }

    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        appendMessage('bot', data.response);

        if (data.journeys && data.journeys.length > 0) {
            appendJourneyCards(data.journeys);
            if (!journeyPlannerToggle.classList.contains('hidden')) {
                journeyPlannerToggle.classList.remove('hidden');
            }
        }

        displayMetadata(data.intent, data.confidence, data.entities || {});
    })
    .catch(err => {
        appendMessage('bot', 'Error: ' + err.message);
        console.error(err);
    });
}

sendButton.addEventListener('click', () => {
    const message = userInput.value.trim();
    if (!message) return;

    userInput.value = '';
    sendChatMessage({ message });
});

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendButton.click();
    }
});

journeyPlannerToggle.addEventListener('click', toggleJourneyPlanner);
planBtn.addEventListener('click', sendPlannedJourney);

document.addEventListener('DOMContentLoaded', () => {
    journeyPlannerToggle.classList.remove('hidden');
});
