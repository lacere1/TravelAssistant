// Chat interface JavaScript
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const intentDisplay = document.getElementById('intentDisplay');
const confidenceDisplay = document.getElementById('confidenceDisplay');
const entitiesDisplay = document.getElementById('entitiesDisplay');

// Sidebar elements (chats + account)
const newChatButton = document.getElementById('newChatButton');
const chatSearchInput = document.getElementById('chatSearchInput');
const chatList = document.getElementById('chatList');
const loginLink = document.getElementById('loginLink');
const logoutButton = document.getElementById('logoutButton');
const accountLoggedOut = document.getElementById('accountLoggedOut');
const accountLoggedIn = document.getElementById('accountLoggedIn');
const currentUsernameEl = document.getElementById('currentUsername');
const accountMenuButton = document.getElementById('accountMenuButton');
const accountDropdown = document.getElementById('accountDropdown');


// Journey planner UI elements
const journeyInputsSection = document.getElementById('journey-inputs');
const journeyPlannerToggle = document.getElementById('journey-planner-toggle');
const fromInput = document.getElementById('from-input');
const toInput = document.getElementById('to-input');
const dateInput = document.getElementById('date-input');
const timeInput = document.getElementById('time-input');
const planBtn = document.getElementById('plan-btn');

// Theme toggle
const themeToggle = document.getElementById('themeToggle');
const themeToggleLabel = document.getElementById('themeToggleLabel');
const THEME_STORAGE_KEY = 'travelAssistantTheme';

let hasShownJourneyInputs = false;
let fromCoord = '';
let toCoord = '';

// Local chat history state (per user, stored in localStorage)
const STORAGE_KEY_BASE = 'travelAssistantChats';
let currentUser = null;
let conversations = [];
let activeChatId = null;
let isRestoringMessages = false;
let currentSearchTerm = '';

function getStorageKey() {
    return currentUser ? `${STORAGE_KEY_BASE}:${currentUser}` : STORAGE_KEY_BASE;
}

function migrateConversation(convo) {
    if (convo.starred === undefined) convo.starred = false;
    if (convo.updatedAt === undefined) {
        convo.updatedAt = convo.messages && convo.messages.length
            ? Date.now()
            : Date.now();
    }
    return convo;
}

function loadConversations() {
    try {
        const raw = localStorage.getItem(getStorageKey());
        const loaded = raw ? JSON.parse(raw) : [];
        conversations = Array.isArray(loaded) ? loaded.map(migrateConversation) : [];
    } catch (e) {
        conversations = [];
        // eslint-disable-next-line no-console
        console.error('Failed to load conversations', e);
    }
}

function saveConversations() {
    try {
        localStorage.setItem(getStorageKey(), JSON.stringify(conversations));
    } catch (e) {
        // eslint-disable-next-line no-console
        console.error('Failed to save conversations', e);
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
    if (diffMins < 60) return `${diffMins} min ago`;
    if (diffHours < 24 && d.getDate() === now.getDate()) return `${diffHours}h ago`;
    if (diffDays === 1 || (diffDays < 2 && d.getDate() !== now.getDate())) return 'Yesterday';
    if (diffDays < 7) return d.toLocaleDateString(undefined, { weekday: 'short' });
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function deleteChat(chatId) {
    conversations = conversations.filter((c) => c.id !== chatId);
    if (activeChatId === chatId) {
        activeChatId = conversations[0]?.id || null;
        if (activeChatId) setActiveChat(activeChatId);
        else if (chatMessages) clearMessages();
    }
    saveConversations();
    renderChatList(currentSearchTerm);
}

    if (wasActive) {
        activeChatId = conversations[0]?.id || null;
        if (activeChatId) setActiveChat(activeChatId);
        else if (chatMessages) clearMessages();
    }
    saveConversations();
    renderChatList(currentSearchTerm);
}

function renameChat(chatId) {
    const convo = conversations.find((c) => c.id === chatId);
    if (!convo) return;
    const newTitle = (window.prompt('Rename chat', convo.title) || '').trim();
    if (newTitle && newTitle !== convo.title) {
        convo.title = newTitle.length > 60 ? newTitle.slice(0, 60) : newTitle;
        saveConversations();
        renderChatList(currentSearchTerm);
    }
}

function getWelcomeMessageHtml() {
    const greeting = currentUser
        ? `Hi ${escapeHtml(currentUser)}, I'm your travel assistant. I can help you with:`
        : "Hi, I'm your travel assistant. I can help you with:";
    return `
        <p>${greeting}</p>
        <ul>
            <li>Rough delay expectations</li>
            <li>Choosing a good time to set off</li>
            <li>Simple route suggestions</li>
        </ul>
        <p>You can ask things like: "How busy is the A1 right now?" or "What time is best to drive into the city?"</p>
    `;
}

function updateWelcomeMessage() {
    if (!chatMessages) return;
    const firstBot = chatMessages.querySelector('.message.bot-message .message-content');
    if (firstBot) firstBot.innerHTML = getWelcomeMessageHtml();
}

function clearMessages() {
    if (!chatMessages) return;
    chatMessages.innerHTML = '';
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = getWelcomeMessageHtml();
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
}

function setActiveChat(chatId) {
    activeChatId = chatId;
    if (!chatMessages) return;
    clearMessages();
    const convo = conversations.find((c) => c.id === chatId);
    if (!convo) {
        highlightActiveChat();
        return;
    }
    isRestoringMessages = true;
    convo.messages.forEach((m) => {
        const p = m.payload;
        if (m.sender === 'bot' && p && (p.timetable || p.disruption)) {
            if (p.timetable) appendTimetableCard(p.timetable, m.text);
            if (p.disruption) appendDisruptionCard(p.disruption, m.text);
        } else {
            addMessage(m.text, m.sender);
            if (p && p.journeys && p.journeys.length) appendJourneyCards(p.journeys, m.text || '', p.journeyOptions || {});
        }
    });
    isRestoringMessages = false;
    highlightActiveChat();
}

function highlightActiveChat() {
    if (!chatList) return;
    Array.from(chatList.children).forEach((item) => {
        if (item.dataset && item.dataset.chatId === activeChatId) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
}

function renderChatList(filterText) {
    if (!chatList) return;
    const term = (filterText || '').toLowerCase();
    let filtered = conversations.filter((c) => !term || (c.title || '').toLowerCase().includes(term));
    const sorted = [...filtered].sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));

    chatList.innerHTML = '';
    sorted.forEach((convo) => {
        const li = document.createElement('li');
        li.className = 'chat-list-item';
        li.dataset.chatId = convo.id;

        const content = document.createElement('div');
        content.className = 'chat-list-item-content';
        content.addEventListener('click', (e) => {
            if (!e.target.closest('.chat-list-item-menu-btn')) setActiveChat(convo.id);
        });
        const titleEl = document.createElement('div');
        titleEl.className = 'chat-list-item-title';
        titleEl.textContent = convo.title || 'Untitled chat';
        const timeEl = document.createElement('div');
        timeEl.className = 'chat-list-item-time';
        timeEl.textContent = formatLastMessageTime(convo.updatedAt);
        content.appendChild(titleEl);
        content.appendChild(timeEl);

        const menuBtn = document.createElement('button');
        menuBtn.type = 'button';
        menuBtn.className = 'chat-list-item-menu-btn';
        menuBtn.setAttribute('aria-label', 'Chat options');
        menuBtn.textContent = '⋮';
        menuBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const menu = li.querySelector('.chat-list-item-menu');
            if (!menu) return;
            document.querySelectorAll('.chat-list-item-menu').forEach((m) => m.classList.add('hidden'));
            const rect = menuBtn.getBoundingClientRect();
            menu.style.left = rect.left + 'px';
            menu.style.top = (rect.bottom + 4) + 'px';
            menu.style.right = 'auto';
            menu.style.bottom = 'auto';
            menu.classList.toggle('hidden');
        });

        const menu = document.createElement('div');
        menu.className = 'chat-list-item-menu hidden';
        const renameBtn = document.createElement('button');
        renameBtn.type = 'button';
        renameBtn.className = 'chat-list-item-menu-item';
        renameBtn.textContent = 'Rename';
        renameBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            menu.classList.add('hidden');
            renameChat(convo.id);
        });
        const deleteBtn = document.createElement('button');
        deleteBtn.type = 'button';
        deleteBtn.className = 'chat-list-item-menu-item chat-list-item-menu-item-danger';
        deleteBtn.textContent = 'Delete';
        deleteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            menu.classList.add('hidden');
            deleteChat(convo.id);
        });
        menu.appendChild(renameBtn);
        menu.appendChild(deleteBtn);

        li.appendChild(content);
        li.appendChild(menuBtn);
        li.appendChild(menu);
        chatList.appendChild(li);
    });

    highlightActiveChat();
}

document.addEventListener('click', (e) => {
    if (!e.target.closest('.chat-list-item-menu') && !e.target.closest('.chat-list-item-menu-btn')) {
        document.querySelectorAll('.chat-list-item-menu').forEach((m) => m.classList.add('hidden'));
    }
});

function addMessageAndStore(text, sender, payload) {
    addMessage(text, sender);
    if (!isRestoringMessages) {
        storeMessage(text, sender, payload);
    }
}

// Handle send button click
sendButton.addEventListener('click', sendMessage);

// Handle Enter key press
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    // Add user message to chat
    addMessageAndStore(message, 'user');
    userInput.value = '';

    // Reveal journey planner toggle after first free-text query
    if (!hasShownJourneyInputs) {
        showJourneyInputsIfNeeded();
    }

    // Show typing indicator
    const typingId = showTypingIndicator();

    // Send message to backend
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message })
    })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator
            removeTypingIndicator(typingId);
            handleChatResponse(data);
        })
        .catch(error => {
            removeTypingIndicator(typingId);
            addMessageAndStore('Sorry, I encountered an error. Please try again.', 'bot');
            console.error('Error:', error);
        });
}

function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    if (sender === 'bot' && text) {
        messageDiv.dataset.messageText = text;
    }

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // Format message text (handle line breaks and lists)
    const formattedText = formatMessage(text);
    contentDiv.innerHTML = formattedText;

    if (sender === 'bot' && text) {
        // No extra per-message controls at this stage.
    }

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatMessage(text) {
    // Convert line breaks to <br>
    text = text.replace(/\n/g, '<br>');
    
    // Convert simple list patterns to HTML lists
    const lines = text.split('<br>');
    let formatted = '';
    let inList = false;
    
    for (let line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('- ') || trimmed.startsWith('• ')) {
            if (!inList) {
                formatted += '<ul>';
                inList = true;
            }
            formatted += '<li>' + trimmed.substring(2) + '</li>';
        } else {
            if (inList) {
                formatted += '</ul>';
                inList = false;
            }
            if (trimmed) {
                formatted += '<p>' + trimmed + '</p>';
            }
        }
    }
    
    if (inList) {
        formatted += '</ul>';
    }
    
    return formatted || '<p>' + text + '</p>';
}

function showTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    messageDiv.id = 'typing-indicator';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = '<span></span><span></span><span></span>';

    contentDiv.appendChild(typingDiv);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);

    chatMessages.scrollTop = chatMessages.scrollHeight;

    return 'typing-indicator';
}

function removeTypingIndicator(id) {
    const indicator = document.getElementById(id);
    if (indicator) {
        indicator.remove();
    }
}

function updateInfoPanel(data) {
    // Update intent
    if (data.intent) {
        intentDisplay.textContent = data.intent.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    // Update confidence
    if (data.confidence !== undefined) {
        const confidencePercent = (data.confidence * 100).toFixed(1);
        confidenceDisplay.textContent = confidencePercent + '%';
        confidenceDisplay.style.color = data.confidence > 0.7 ? '#28a745' : data.confidence > 0.5 ? '#ffc107' : '#dc3545';
    }

    // Update entities
    if (data.entities && Object.keys(data.entities).length > 0) {
        const entitiesList = Object.entries(data.entities)
            .map(([key, value]) => `${key}: ${value}`)
            .join(', ');
        entitiesDisplay.textContent = entitiesList;
    } else {
        entitiesDisplay.textContent = 'None detected';
    }
}

// -------- Journey planner helpers (cards, toggle, structured planning) -----

function updateJourneyPlannerToggle(expanded) {
    if (!journeyPlannerToggle || !journeyInputsSection) return;
    const isExpanded = expanded ?? !journeyInputsSection.classList.contains('hidden');
    journeyPlannerToggle.setAttribute('aria-expanded', String(isExpanded));
    journeyPlannerToggle.textContent = isExpanded ? 'Hide journey planner' : 'Show journey planner';
}

function toggleJourneyPlanner() {
    if (!journeyInputsSection) return;
    journeyInputsSection.classList.toggle('hidden');
    updateJourneyPlannerToggle();
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function getLegIconClass(mode) {
    const m = (mode || '').toLowerCase();
    if (m === 'walk' || m === 'walking') return 'leg-icon walk';
    if (m === 'bus') return 'leg-icon bus';
    if (m === 'tube' || m === 'metro' || m === 'rail' || m === 'dlr' || m === 'overground' || m === 'tram') return 'leg-icon tube';
    return 'leg-icon walk';
}

function getLegLineClass(mode) {
    const m = (mode || '').toLowerCase();
    if (m === 'walk' || m === 'walking') return 'leg-line leg-line-walk';
    if (m === 'bus') return 'leg-line leg-line-bus';
    return 'leg-line leg-line-tube';
}

function tryParseJson(str) {
    try {
        return typeof str === 'string' ? JSON.parse(str) : str;
    } catch {
        return [];
    }
}

function buildDirectionsHtml(steps) {
    if (!Array.isArray(steps) || steps.length === 0) return '';
    const list = document.createElement('div');
    list.className = 'journey-leg-steps';
    steps.forEach((step) => {
        const desc = step.description || step.detailedDescription || step.turnInstruction || '';
        const dist = step.distance != null ? step.distance + 'm' : '';
        const line = document.createElement('div');
        line.className = 'journey-leg-step';
        line.innerHTML = `<span class="journey-step-desc">${escapeHtml(desc)}</span>${dist ? `<span class="journey-step-dist">${escapeHtml(String(dist))}</span>` : ''}`;
        line.appendChild(document.createTextNode(''));
        list.appendChild(line);
    });
    return list.outerHTML;
}

function buildStopsHtml(stops) {
    if (!Array.isArray(stops) || stops.length === 0) return '';
    const list = document.createElement('ul');
    list.className = 'journey-leg-stops';
    stops.forEach((name) => {
        const li = document.createElement('li');
        li.textContent = name;
        list.appendChild(li);
    });
    return list.outerHTML;
}

function buildJourneyCard(journey) {
    const card = document.createElement('div');
    card.className = 'journey-card';

    const farePounds = journey.fare_pence != null ? (journey.fare_pence / 100).toFixed(2) : null;

    const top = document.createElement('div');
    top.className = 'journey-card-top';
    top.innerHTML = `
    <div class="journey-card-times">
      <span class="journey-card-time-range">${journey.departure} – ${journey.arrival}</span>
      ${farePounds != null ? `<span class="journey-card-fare">£${farePounds} off peak</span>` : ''}
    </div>
    <div class="journey-card-duration"><strong>${journey.duration}</strong> mins</div>
  `;
    card.appendChild(top);

    const timeline = document.createElement('div');
    timeline.className = 'journey-card-timeline';
    journey.legs.forEach((leg, i) => {
        const isLast = i === journey.legs.length - 1;
        const lineClass = getLegLineClass(leg.mode);
        const iconClass = getLegIconClass(leg.mode);
        const legMins = leg.duration != null ? leg.duration : '';
        const m = (leg.mode || '').toLowerCase();
        const isWalk = m === 'walk' || m === 'walking';
        const hasDirections = isWalk && (leg.steps?.length || (leg.fromLatLng && leg.toLatLng));
        const hasStops = !isWalk && leg.stops?.length;
        const showLink = hasDirections || hasStops;
        const linkText = isWalk ? 'View directions' : 'View stops';
        const row = document.createElement('div');
        row.className = 'journey-card-leg';
        row.dataset.legIndex = String(i);
        row.dataset.isWalk = isWalk ? '1' : '0';
        if (hasDirections) {
            row.dataset.steps = JSON.stringify(leg.steps || []);
            if (leg.fromLatLng) row.dataset.fromLatLng = leg.fromLatLng;
            if (leg.toLatLng) row.dataset.toLatLng = leg.toLatLng;
        }
        if (hasStops) row.dataset.stops = JSON.stringify(leg.stops);
        row.innerHTML = `
      <div class="journey-card-leg-left">
        <div class="${iconClass}"></div>
        ${!isLast ? `<div class="${lineClass}"></div>` : ''}
      </div>
      <div class="journey-card-leg-right">
        <div class="journey-card-leg-detail">${leg.detail || leg.mode}</div>
        <div class="journey-card-leg-meta">
          ${legMins !== '' ? `${legMins} min` : ''}
          ${showLink ? `<a href="#" class="journey-leg-link" data-link-type="${isWalk ? 'directions' : 'stops'}">${linkText}</a>` : ''}
        </div>
        <div class="journey-leg-expandable hidden" aria-live="polite"></div>
      </div>
    `;
        timeline.appendChild(row);
    });

    const lastLeg = journey.legs.length ? journey.legs[journey.legs.length - 1] : null;
    const destLabel = lastLeg && lastLeg.detail ? lastLeg.detail.replace(/^Walk to\s+/i, '').trim() || 'Destination' : 'Destination';
    const destRow = document.createElement('div');
    destRow.className = 'journey-card-leg journey-card-dest';
    destRow.innerHTML = `
    <div class="journey-card-leg-left">
      <div class="leg-icon dest"></div>
    </div>
    <div class="journey-card-leg-right">
      <div class="journey-card-leg-detail">${destLabel}</div>
    </div>
  `;
    timeline.appendChild(destRow);
    card.appendChild(timeline);

    const actions = document.createElement('div');
    actions.className = 'journey-card-actions';
    actions.innerHTML = '<button type="button" class="journey-card-btn">View details</button><button type="button" class="journey-card-btn">Map view</button>';
    card.appendChild(actions);

    return card;
}

function appendJourneyCards(journeys, debugText, options) {
    options = options || {};
    const tflUrl = options.tflJourneyUrl || '';
    const fromId = options.fromId || '';
    const toId = options.toId || '';
    let mapUrl = '';
    if (fromId && toId) {
        mapUrl = 'https://www.google.com/maps/dir/?api=1&origin=' + encodeURIComponent(fromId) + '&destination=' + encodeURIComponent(toId) + '&travelmode=transit';
    } else {
        mapUrl = 'https://www.google.com/maps';
    }

    const wrap = document.createElement('div');
    wrap.className = 'message bot-message journey-cards-wrap';
    wrap.dataset.tflUrl = tflUrl;
    wrap.dataset.mapUrl = mapUrl;
    journeys.forEach((j) => wrap.appendChild(buildJourneyCard(j)));

    wrap.addEventListener('click', (e) => {
        const link = e.target.closest('.journey-leg-link');
        if (link) {
            e.preventDefault();
            const legRow = link.closest('.journey-card-leg');
            const expandable = legRow?.querySelector('.journey-leg-expandable');
            const type = link.dataset.linkType;

            if (type === 'directions') {
                const stepsJson = legRow?.dataset.steps;
                const steps = stepsJson ? tryParseJson(stepsJson) : [];
                const fromLatLng = legRow?.dataset.fromLatLng;
                const toLatLng = legRow?.dataset.toLatLng;

                if (expandable?.classList.contains('hidden')) {
                    if (steps?.length) {
                        expandable.innerHTML = buildDirectionsHtml(steps);
                        expandable.classList.remove('hidden');
                        link.textContent = 'Hide directions';
                    } else if (fromLatLng && toLatLng) {
                        const url = 'https://www.google.com/maps/dir/?api=1&origin=' + encodeURIComponent(fromLatLng) + '&destination=' + encodeURIComponent(toLatLng) + '&travelmode=walking';
                        window.open(url, '_blank');
                    } else {
                        const w = link.closest('.journey-cards-wrap');
                        if (w?.dataset?.mapUrl) window.open(w.dataset.mapUrl, '_blank');
                    }
                } else {
                    expandable.classList.add('hidden');
                    expandable.innerHTML = '';
                    link.textContent = 'View directions';
                }
                return;
            }

            if (type === 'stops') {
                const stopsJson = legRow?.dataset.stops;
                const stops = stopsJson ? tryParseJson(stopsJson) : [];
                if (expandable?.classList.contains('hidden')) {
                    if (stops?.length) {
                        expandable.innerHTML = buildStopsHtml(stops);
                        expandable.classList.remove('hidden');
                        link.textContent = 'Hide stops';
                    }
                } else {
                    expandable.classList.add('hidden');
                    expandable.innerHTML = '';
                    link.textContent = 'View stops';
                }
                return;
            }
        }

        const btn = e.target.closest('.journey-card-btn');
        if (!btn) return;
        const actions = btn.closest('.journey-card-actions');
        const idx = Array.prototype.indexOf.call(actions.children, btn);
        const w = btn.closest('.journey-cards-wrap');
        if (idx === 0 && w?.dataset?.tflUrl) window.open(w.dataset.tflUrl, '_blank');
        else if (idx === 1 && w?.dataset?.mapUrl) window.open(w.dataset.mapUrl, '_blank');
    });

    if (debugText && debugText.includes('[debug]')) {
        const debugLine = debugText.split('\n').find((line) => line.includes('[debug]')) || debugText;
        const debug = document.createElement('div');
        debug.className = 'journey-debug';
        debug.textContent = debugLine.trim();
        wrap.appendChild(debug);
    }
    chatMessages.appendChild(wrap);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showJourneyInputsIfNeeded() {
    if (hasShownJourneyInputs) return;
    hasShownJourneyInputs = true;
    if (journeyPlannerToggle) journeyPlannerToggle.classList.remove('hidden');
}

function appendTimetableCard(timetableData, messageText) {
    if (!timetableData || !chatMessages) return;
    const stopName = timetableData.stop_name || 'Stop';
    const busArrivals = timetableData.bus_arrivals || [];
    const trainArrivals = timetableData.train_arrivals || [];
    const busGrouped = timetableData.bus_arrivals_grouped || {};
    const trainByDir = timetableData.train_arrivals_by_direction || {};

    const wrap = document.createElement('div');
    wrap.className = 'message bot-message timetable-cards-wrap';

    const header = document.createElement('div');
    header.className = 'timetable-card-header';
    header.innerHTML = `<span class="timetable-card-header-title">${escapeHtml(stopName)}</span><span class="timetable-card-header-subtitle">Next</span>`;
    wrap.appendChild(header);

    const card = document.createElement('div');
    card.className = 'timetable-card';

    const directionOrder = ['Northbound', 'Southbound', 'Eastbound', 'Westbound', 'Clockwise', 'Anticlockwise', 'Unknown'];

    function addRow(line, destination, timeLabel, isTrain) {
        const row = document.createElement('div');
        row.className = 'timetable-row';
        const icon = isTrain ? 'tube' : 'bus';
        row.innerHTML = `
            <span class="timetable-row-icon timetable-icon-${icon}" aria-hidden="true"></span>
            <div class="timetable-row-info">
                <span class="timetable-row-line">${escapeHtml(String(line))}</span>
                <span class="timetable-row-dest">${escapeHtml(destination || '')}</span>
            </div>
            <div class="timetable-row-time">${escapeHtml(timeLabel)}</div>
        `;
        card.appendChild(row);
    }

    if (Object.keys(busGrouped).length > 0) {
        Object.keys(busGrouped).forEach((gid) => {
            const g = busGrouped[gid];
            const groupName = g.group_name || stopName;
            const stops = g.stops || {};
            Object.keys(stops).forEach((stopLabel) => {
                const arrivals = stops[stopLabel] || [];
                const towards = stopLabel !== 'Stop' && stopLabel !== groupName ? stopLabel : '';
                arrivals.slice(0, 8).forEach((b) => {
                    const timeLabel = b.time_minutes != null ? `${Math.round(Number(b.time_minutes))} min` : '–';
                    addRow(b.line || '–', b.destination || towards || '–', timeLabel, false);
                });
            });
        });
    } else if (busArrivals.length > 0) {
        busArrivals.slice(0, 10).forEach((b) => {
            const timeLabel = b.time_minutes != null ? `${Math.round(Number(b.time_minutes))} min` : '–';
            addRow(b.line || '–', b.destination || '–', timeLabel, false);
        });
    }

    if (Object.keys(trainByDir).length > 0) {
        const sortedDirs = Object.keys(trainByDir).sort((a, b) => {
            const i = directionOrder.indexOf(a);
            const j = directionOrder.indexOf(b);
            return (i === -1 ? 999 : i) - (j === -1 ? 999 : j);
        });
        sortedDirs.forEach((direction) => {
            const dirTrains = trainByDir[direction] || [];
            const dirHeader = document.createElement('div');
            dirHeader.className = 'timetable-direction-header';
            dirHeader.textContent = direction;
            card.appendChild(dirHeader);
            dirTrains.slice(0, 6).forEach((t) => {
                const timeLabel = t.time_minutes != null ? `${Math.round(Number(t.time_minutes))} min` : '–';
                const dest = t.destination || '–';
                const platform = t.platform_number || (t.platform ? String(t.platform).replace(/platform\s*/i, '') : '');
                const destStr = platform ? `${dest} (Platform ${platform})` : dest;
                addRow(t.line || '–', destStr, timeLabel, true);
            });
        });
    } else if (trainArrivals.length > 0) {
        trainArrivals.slice(0, 10).forEach((t) => {
            const timeLabel = t.time_minutes != null ? `${Math.round(Number(t.time_minutes))} min` : '–';
            addRow(t.line || '–', t.destination || '–', timeLabel, true);
        });
    }

    if (card.children.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'timetable-empty';
        empty.textContent = 'No arrivals in the near future.';
        card.appendChild(empty);
    }

    wrap.appendChild(card);
    chatMessages.appendChild(wrap);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendDisruptionCard(disruptionData, messageText) {
    if (!disruptionData || !chatMessages) return;
    const lineName = disruptionData.line || disruptionData.route || 'Service';
    const status = disruptionData.status || 'Status unavailable';
    const description = disruptionData.description || '';
    const affected = disruptionData.affected_locations || [];
    const alternatives = disruptionData.alternatives || [];

    const wrap = document.createElement('div');
    wrap.className = 'message bot-message disruption-cards-wrap';

    const header = document.createElement('div');
    header.className = 'disruption-card-header';
    header.innerHTML = `<span class="disruption-card-header-title">${escapeHtml(lineName)}${lineName.toLowerCase().indexOf('line') === -1 && !/^\d+$/.test(lineName) ? ' Line' : ''}</span>`;
    wrap.appendChild(header);

    const card = document.createElement('div');
    card.className = 'disruption-card';

    const statusRow = document.createElement('div');
    statusRow.className = 'disruption-status-row';
    const statusClass = status.toLowerCase().indexOf('good') !== -1 ? 'disruption-status-good' : 'disruption-status-issue';
    statusRow.innerHTML = `<span class="disruption-status-label ${statusClass}">${escapeHtml(status)}</span>`;
    card.appendChild(statusRow);

    if (description) {
        const descEl = document.createElement('div');
        descEl.className = 'disruption-description';
        descEl.textContent = description;
        card.appendChild(descEl);
    }
    if (affected.length > 0) {
        const affEl = document.createElement('div');
        affEl.className = 'disruption-affected';
        affEl.innerHTML = '<span class="disruption-affected-label">Affected:</span> ' + affected.map((a) => escapeHtml(a)).join('; ');
        card.appendChild(affEl);
    }
    if (alternatives.length > 0) {
        const altEl = document.createElement('div');
        altEl.className = 'disruption-alternatives';
        altEl.innerHTML = '<span class="disruption-alt-label">Alternatives:</span> ' + alternatives.map((a) => escapeHtml(a)).join('; ');
        card.appendChild(altEl);
    }

    wrap.appendChild(card);
    chatMessages.appendChild(wrap);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function handleChatResponse(data) {
    if (data.error) {
        addMessageAndStore('Sorry, I encountered an error: ' + data.error, 'bot');
        return;
    }

    if (data.journeys && data.journeys.length > 0) {
        const journeyOptions = {
            tflJourneyUrl: data.tfl_journey_url || '',
            fromId: data.from_id || '',
            toId: data.to_id || '',
        };
        addMessageAndStore(data.response || '', 'bot', { journeys: data.journeys, journeyOptions });
        appendJourneyCards(data.journeys, data.response || '', journeyOptions);
        showJourneyInputsIfNeeded();
        if (journeyInputsSection && journeyInputsSection.classList.contains('hidden')) {
            journeyInputsSection.classList.remove('hidden');
            updateJourneyPlannerToggle(true);
        }
    } else if (data.timetable || data.disruption) {
        const payload = {};
        if (data.timetable) payload.timetable = data.timetable;
        if (data.disruption) payload.disruption = data.disruption;
        storeMessage(data.response || '', 'bot', payload);
        if (data.timetable) appendTimetableCard(data.timetable, data.response);
        if (data.disruption) appendDisruptionCard(data.disruption, data.response);
    } else if (data.response) {
        addMessageAndStore(data.response, 'bot');
        if (data.disambiguation) {
            showJourneyInputsIfNeeded();
            if (journeyInputsSection && journeyInputsSection.classList.contains('hidden')) {
                journeyInputsSection.classList.remove('hidden');
                updateJourneyPlannerToggle(true);
            }
        }
    }

    updateInfoPanel(data);
}

function sendPlannedJourney() {
    if (!fromInput || !toInput) return;
    const from = fromInput.value.trim();
    const to = toInput.value.trim();
    const date = dateInput ? dateInput.value : '';
    const time = timeInput ? timeInput.value : '';
    const fromId = fromCoord || '';
    const toId = toCoord || '';

    if (!from || !to) {
        addMessageAndStore('Please choose both a start and destination before planning.', 'bot');
        return;
    }

    let whenPhrase = '';
    if (date && time) {
        whenPhrase = ` on ${date} at ${time}`;
    } else if (time) {
        whenPhrase = ` at ${time}`;
    }

    const query = `Plan a journey from ${from} to ${to}${whenPhrase}`;
    addMessageAndStore(query, 'user');

    const typingId = showTypingIndicator();

    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: query, from, to, date, time, fromId, toId }),
    })
        .then((res) => res.json())
        .then((data) => {
            removeTypingIndicator(typingId);
            handleChatResponse(data);
        })
        .catch(() => {
            removeTypingIndicator(typingId);
            addMessageAndStore('Sorry, something went wrong talking to the server.', 'bot');
        });
}

// Wire up journey planner controls if present
if (journeyPlannerToggle) {
    journeyPlannerToggle.addEventListener('click', toggleJourneyPlanner);
}
if (planBtn) {
    planBtn.addEventListener('click', sendPlannedJourney);
}

// Sidebar actions
if (newChatButton) {
    newChatButton.addEventListener('click', () => {
        activeChatId = null;
        clearMessages();
        highlightActiveChat();
    });
}

if (chatSearchInput) {
    chatSearchInput.addEventListener('input', (e) => {
        currentSearchTerm = e.target.value || '';
        renderChatList(currentSearchTerm);
    });
}

function setLoggedIn(username) {
    currentUser = username || null;
    if (accountLoggedOut && accountLoggedIn && currentUsernameEl) {
        accountLoggedOut.classList.add('hidden');
        accountLoggedIn.classList.remove('hidden');
        currentUsernameEl.textContent = currentUser || '';
    }
    loadConversations();
    activeChatId = conversations[0]?.id || null;
    renderChatList(currentSearchTerm);
    if (activeChatId) {
        setActiveChat(activeChatId);
    } else {
        clearMessages();
    }
}

function setLoggedOut() {
    currentUser = null;
    if (accountLoggedOut && accountLoggedIn && currentUsernameEl) {
        accountLoggedOut.classList.remove('hidden');
        accountLoggedIn.classList.add('hidden');
        currentUsernameEl.textContent = '';
    }
    loadConversations();
    activeChatId = conversations[0]?.id || null;
    renderChatList(currentSearchTerm);
    clearMessages();
}

function closeAccountDropdown() {
    if (accountDropdown) accountDropdown.classList.add('hidden');
    if (accountMenuButton) accountMenuButton.setAttribute('aria-expanded', 'false');
}

function toggleAccountDropdown() {
    if (!accountDropdown || !accountMenuButton) return;
    const isOpen = !accountDropdown.classList.contains('hidden');
    if (isOpen) {
        accountDropdown.classList.add('hidden');
        accountMenuButton.setAttribute('aria-expanded', 'false');
    } else {
        accountDropdown.classList.remove('hidden');
        accountMenuButton.setAttribute('aria-expanded', 'true');
    }
}

if (accountMenuButton) {
    accountMenuButton.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleAccountDropdown();
    });
}

document.addEventListener('click', (e) => {
    if (accountDropdown && !accountDropdown.classList.contains('hidden')) {
        const wrap = document.querySelector('.account-button-wrap');
        if (wrap && !wrap.contains(e.target)) closeAccountDropdown();
    }
});

if (accountDropdown) {
    accountDropdown.addEventListener('click', (e) => e.stopPropagation());
}

if (logoutButton) {
    logoutButton.addEventListener('click', () => {
        closeAccountDropdown();
        fetch('/logout', { method: 'POST' })
            .finally(() => {
                setLoggedOut();
            });
    });
}

function getTheme() {
    try {
        const stored = localStorage.getItem(THEME_STORAGE_KEY);
        if (stored === 'dark' || stored === 'light') return stored;
    } catch (e) {
        // ignore
    }
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    if (themeToggleLabel) {
        themeToggleLabel.textContent = theme === 'dark' ? 'Dark' : 'Light';
    }
    try {
        localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch (e) {
        // ignore
    }
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    applyTheme(current === 'light' ? 'dark' : 'light');
}

// Initialize on load
window.addEventListener('load', () => {
    applyTheme(getTheme());
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
    if (userInput) {
        userInput.focus();
    }

    fetch('/me')
        .then((res) => res.json())
        .then((data) => {
            if (data.username) {
                setLoggedIn(data.username);
            } else {
                setLoggedOut();
            }
            updateWelcomeMessage();
        })
        .catch(() => {
            setLoggedOut();
            updateWelcomeMessage();
        });
});
