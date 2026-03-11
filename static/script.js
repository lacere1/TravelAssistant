const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const voiceInputButton = document.getElementById('voiceInputButton');
const intentDisplay = document.getElementById('intentDisplay');
const confidenceDisplay = document.getElementById('confidenceDisplay');
const entitiesDisplay = document.getElementById('entitiesDisplay');

const newChatButton = document.getElementById('newChatButton');
const chatSearchInput = document.getElementById('chatSearchInput');
const chatList = document.getElementById('chatList');
const chatFilterAll = document.getElementById('chatFilterAll');
const chatFilterStarred = document.getElementById('chatFilterStarred');
const loginLink = document.getElementById('loginLink');
const logoutButton = document.getElementById('logoutButton');
const accountLoggedOut = document.getElementById('accountLoggedOut');
const accountLoggedIn = document.getElementById('accountLoggedIn');
const currentUsernameEl = document.getElementById('currentUsername');
const accountMenuButton = document.getElementById('accountMenuButton');
const accountDropdown = document.getElementById('accountDropdown');
const themeToggle = document.getElementById('themeToggle');
const themeToggleLabel = document.getElementById('themeToggleLabel');

const journeyInputsSection = document.getElementById('journey-inputs');
const journeyPlannerToggle = document.getElementById('journey-planner-toggle');
const fromInput = document.getElementById('from-input');
const toInput = document.getElementById('to-input');
const dateInput = document.getElementById('date-input');
const timeInput = document.getElementById('time-input');
const planBtn = document.getElementById('plan-btn');

const STORAGE_KEY_BASE = 'travelAssistantChats';
const THEME_STORAGE_KEY = 'travelAssistantTheme';
let currentUser = null;
let conversations = [];
let activeChatId = null;
let currentSearchTerm = '';
let showStarredOnly = false;
let selectedChatIds = new Set();
let isListening = false;

function getStorageKey() {
    return currentUser ? `${STORAGE_KEY_BASE}:${currentUser}` : STORAGE_KEY_BASE;
}

function migrateConversation(convo) {
    if (convo.starred === undefined) convo.starred = false;
    if (convo.updatedAt === undefined) convo.updatedAt = Date.now();
    return convo;
}

function loadConversations() {
    selectedChatIds.clear();
    try {
        const raw = localStorage.getItem(getStorageKey());
        const loaded = raw ? JSON.parse(raw) : [];
        conversations = Array.isArray(loaded) ? loaded.map(migrateConversation) : [];
    } catch (e) {
        conversations = [];
    }
}

function saveConversations() {
    try {
        localStorage.setItem(getStorageKey(), JSON.stringify(conversations));
    } catch (e) {
    }
}

function ensureActiveConversation(initialText) {
    if (activeChatId) return;
    const id = `chat-${Date.now()}`;
    const titleSource = (initialText || '').trim() || 'New chat';
    const title = titleSource.length > 40 ? `${titleSource.slice(0, 40)}…` : titleSource;
    const convo = { id, title, messages: [], starred: false, updatedAt: Date.now() };
    conversations.unshift(convo);
    activeChatId = id;
}

function storeMessage(text, sender, payload) {
    ensureActiveConversation(sender === 'user' ? text : '');
    const convo = conversations.find((c) => c.id === activeChatId);
    if (!convo) return;
    const msg = { sender, text };
    if (payload && typeof payload === 'object' && Object.keys(payload).length > 0) {
        msg.payload = payload;
    }
    convo.messages.push(msg);
    convo.updatedAt = Date.now();
    if (convo.messages.length === 1 && sender === 'user') {
        const t = text.trim();
        convo.title = t.length > 40 ? `${t.slice(0, 40)}…` : t || convo.title;
    }
    saveConversations();
    renderChatList(currentSearchTerm);
}

function formatLastMessageTime(updatedAt) {
    if (!updatedAt) return '';
    const d = new Date(updatedAt);
    const now = new Date();
    const diffMs = now - d;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m`;
    if (diffHours < 24 && d.getDate() === now.getDate()) return `${diffHours}h`;
    if (diffDays === 1 || (diffDays < 2 && d.getDate() !== now.getDate())) return 'Yday';
    if (diffDays < 7) return d.toLocaleDateString(undefined, { weekday: 'short' });
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function deleteChat(chatId) {
    conversations = conversations.filter((c) => c.id !== chatId);
    selectedChatIds.delete(chatId);
    if (activeChatId === chatId) {
        activeChatId = conversations[0]?.id || null;
        if (activeChatId) setActiveChat(activeChatId);
        else clearMessages();
    }
    saveConversations();
    renderChatList(currentSearchTerm);
}

function toggleStar(chatId) {
    const convo = conversations.find((c) => c.id === chatId);
    if (!convo) return;
    convo.starred = !convo.starred;
    saveConversations();
    renderChatList(currentSearchTerm);
}

function toggleChatSelected(chatId) {
    if (selectedChatIds.has(chatId)) selectedChatIds.delete(chatId);
    else selectedChatIds.add(chatId);
    renderChatList(currentSearchTerm);
}

function renderChatList(searchTerm = '') {
    chatList.innerHTML = '';
    const filtered = conversations.filter((c) => {
        const matchesSearch = c.title.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesFilter = !showStarredOnly || c.starred;
        return matchesSearch && matchesFilter;
    });

    filtered.forEach((convo) => {
        const li = document.createElement('li');
        li.className = 'chat-list-item';
        if (activeChatId === convo.id) li.classList.add('active');

        const label = document.createElement('label');
        label.className = 'chat-list-item-label';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'chat-list-item-checkbox';
        checkbox.checked = selectedChatIds.has(convo.id);
        checkbox.addEventListener('change', () => toggleChatSelected(convo.id));

        const titleButton = document.createElement('button');
        titleButton.type = 'button';
        titleButton.className = 'chat-list-item-title';
        titleButton.textContent = convo.title;
        titleButton.addEventListener('click', () => setActiveChat(convo.id));

        const timeSpan = document.createElement('span');
        timeSpan.className = 'chat-list-item-time';
        timeSpan.textContent = formatLastMessageTime(convo.updatedAt);

        const starBtn = document.createElement('button');
        starBtn.type = 'button';
        starBtn.className = 'chat-list-item-star';
        starBtn.textContent = convo.starred ? '★' : '☆';
        starBtn.addEventListener('click', () => toggleStar(convo.id));

        const deleteBtn = document.createElement('button');
        deleteBtn.type = 'button';
        deleteBtn.className = 'chat-list-item-delete';
        deleteBtn.textContent = '✕';
        deleteBtn.addEventListener('click', () => deleteChat(convo.id));

        label.appendChild(checkbox);
        label.appendChild(titleButton);
        li.appendChild(label);
        li.appendChild(timeSpan);
        li.appendChild(starBtn);
        li.appendChild(deleteBtn);
        chatList.appendChild(li);
    });
}

function setActiveChat(chatId) {
    activeChatId = chatId;
    const convo = conversations.find((c) => c.id === chatId);
    clearMessages();
    if (convo && convo.messages) {
        convo.messages.forEach((msg) => {
            appendMessage(msg.sender, msg.text);
        });
    }
    renderChatList(currentSearchTerm);
    displayMetadata('', 0, {});
}

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
        to: to,
        date: dateInput.value || undefined,
        time: timeInput.value || undefined
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
        storeMessage(message, 'user', {});
    }

    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        appendMessage('bot', data.response);
        storeMessage(data.response, 'bot', {});

        if (data.journeys && data.journeys.length > 0) {
            appendJourneyCards(data.journeys);
        }

        displayMetadata(data.intent, data.confidence, data.entities || {});
    })
    .catch(err => {
        appendMessage('bot', 'Error: ' + err.message);
        console.error(err);
    });
}

function startVoiceInput() {
    if (isListening) return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        alert('Voice input not supported in this browser');
        return;
    }

    isListening = true;
    voiceInputButton.classList.add('listening');
    voiceInputButton.textContent = 'Listening...';

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
        }
        userInput.value = transcript.trim();
        isListening = false;
        voiceInputButton.classList.remove('listening');
        voiceInputButton.textContent = '🎤 Voice input';
    };

    recognition.onerror = () => {
        isListening = false;
        voiceInputButton.classList.remove('listening');
        voiceInputButton.textContent = '🎤 Voice input';
    };

    recognition.onend = () => {
        isListening = false;
        voiceInputButton.classList.remove('listening');
        voiceInputButton.textContent = '🎤 Voice input';
    };

    recognition.start();
}

function getTheme() {
    return localStorage.getItem(THEME_STORAGE_KEY) || 'light';
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_STORAGE_KEY, theme);
    const label = themeToggleLabel.textContent;
    if (theme === 'dark') {
        themeToggleLabel.textContent = 'Dark';
    } else {
        themeToggleLabel.textContent = 'Light';
    }
}

function toggleTheme() {
    const current = getTheme();
    const next = current === 'dark' ? 'light' : 'dark';
    applyTheme(next);
}

function checkLoggedIn() {
    return fetch('/me')
        .then(res => res.json())
        .then(data => {
            if (data.username) {
                currentUser = data.username;
                loadConversations();
                renderChatList();
                accountLoggedOut.classList.add('hidden');
                accountLoggedIn.classList.remove('hidden');
                currentUsernameEl.textContent = currentUser;
                return true;
            } else {
                accountLoggedIn.classList.add('hidden');
                accountLoggedOut.classList.remove('hidden');
                return false;
            }
        })
        .catch(() => false);
}

newChatButton.addEventListener('click', () => {
    activeChatId = null;
    clearMessages();
    renderChatList();
});

chatSearchInput.addEventListener('input', (e) => {
    currentSearchTerm = e.target.value;
    renderChatList(currentSearchTerm);
});

chatFilterAll.addEventListener('click', () => {
    showStarredOnly = false;
    chatFilterAll.classList.add('active');
    chatFilterStarred.classList.remove('active');
    renderChatList(currentSearchTerm);
});

chatFilterStarred.addEventListener('click', () => {
    showStarredOnly = true;
    chatFilterStarred.classList.add('active');
    chatFilterAll.classList.remove('active');
    renderChatList(currentSearchTerm);
});

logoutButton.addEventListener('click', () => {
    fetch('/logout', { method: 'POST' })
        .then(() => {
            currentUser = null;
            conversations = [];
            activeChatId = null;
            clearMessages();
            accountLoggedIn.classList.add('hidden');
            accountLoggedOut.classList.remove('hidden');
            chatList.innerHTML = '';
        });
});

accountMenuButton.addEventListener('click', () => {
    const isHidden = accountDropdown.classList.contains('hidden');
    if (isHidden) {
        accountDropdown.classList.remove('hidden');
        accountMenuButton.setAttribute('aria-expanded', 'true');
    } else {
        accountDropdown.classList.add('hidden');
        accountMenuButton.setAttribute('aria-expanded', 'false');
    }
});

document.addEventListener('click', (e) => {
    if (!accountMenuButton.contains(e.target) && !accountDropdown.contains(e.target)) {
        accountDropdown.classList.add('hidden');
        accountMenuButton.setAttribute('aria-expanded', 'false');
    }
});

themeToggle.addEventListener('click', toggleTheme);

voiceInputButton.addEventListener('click', startVoiceInput);

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
    applyTheme(getTheme());
    checkLoggedIn();
    journeyPlannerToggle.classList.remove('hidden');
});
