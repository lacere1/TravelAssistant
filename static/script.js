// Chat interface JavaScript
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const voiceInputButton = document.getElementById('voiceInputButton');
const intentDisplay = document.getElementById('intentDisplay');
const confidenceDisplay = document.getElementById('confidenceDisplay');
const entitiesDisplay = document.getElementById('entitiesDisplay');

// Sidebar elements (chats + account)
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
const accountMenuAvatar = document.getElementById('accountMenuAvatar');
const changeAvatarButton = document.getElementById('changeAvatarButton');
const removeAvatarButton = document.getElementById('removeAvatarButton');
const avatarFileInput = document.getElementById('avatarFileInput');

const AVATAR_STORAGE_KEY_PREFIX = 'travelAssistantAvatar:';

// ---------------------------------------------------------------------------
// Journey Disambiguation: Geolocation + User History (localStorage)
// ---------------------------------------------------------------------------

const JOURNEY_HISTORY_KEY = 'travelAssistantJourneyHistory';
const TEST_LOCATION_KEY = 'travelAssistantTestLocation';

/** Current user geolocation (updated after permission + in background). */
let userGeoLat = null;
let userGeoLon = null;

/** Optional { lat, lon } — when set, sent instead of real GPS (local testing). */
let testLocationOverride = null;

function loadTestLocationFromStorage() {
    try {
        const raw = localStorage.getItem(TEST_LOCATION_KEY);
        if (!raw) return null;
        const o = JSON.parse(raw);
        if (
            o &&
            typeof o.lat === 'number' &&
            typeof o.lon === 'number' &&
            !Number.isNaN(o.lat) &&
            !Number.isNaN(o.lon)
        ) {
            return { lat: o.lat, lon: o.lon };
        }
    } catch (e) {
        /* ignore */
    }
    return null;
}

function saveTestLocationToStorage(loc) {
    if (!loc) {
        localStorage.removeItem(TEST_LOCATION_KEY);
        return;
    }
    try {
        localStorage.setItem(TEST_LOCATION_KEY, JSON.stringify({ lat: loc.lat, lon: loc.lon }));
    } catch (e) {
        console.warn('[TestLocation] Failed to save:', e);
    }
}

function clearTestLocationOverride() {
    testLocationOverride = null;
    saveTestLocationToStorage(null);
    syncTestLocationPanel();
    updateLocationButtonUI();
}

/**
 * Parse "51.5, -0.12" or "51.5 -0.12" into { lat, lon }.
 */
function parseLatLonPair(text) {
    const t = (text || '').trim();
    if (!t) return null;
    const parts = t.split(/[\s,]+/).filter(Boolean);
    if (parts.length < 2) return null;
    const lat = parseFloat(parts[0]);
    const lon = parseFloat(parts[1]);
    if (Number.isNaN(lat) || Number.isNaN(lon)) return null;
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return null;
    return { lat, lon };
}

function setTestLocationOverride(loc) {
    testLocationOverride = loc;
    saveTestLocationToStorage(loc);
    syncTestLocationPanel();
    updateLocationButtonUI();
}

function syncTestLocationPanel() {
    const status = document.getElementById('testLocationStatus');
    const manual = document.getElementById('testLocationManual');
    const preset = document.getElementById('testLocationPreset');
    if (!status) return;

    if (testLocationOverride) {
        const { lat, lon } = testLocationOverride;
        status.textContent = `Using test: ${lat.toFixed(5)}, ${lon.toFixed(5)} (overrides GPS)`;
        if (manual && !manual.matches(':focus')) {
            manual.value = `${lat}, ${lon}`;
        }
        if (preset) {
            let matched = '';
            Array.from(preset.options).forEach((o) => {
                if (!o.value) return;
                const p = parseLatLonPair(o.value.replace(',', ' '));
                if (p && Math.abs(p.lat - lat) < 0.0002 && Math.abs(p.lon - lon) < 0.0002) {
                    matched = o.value;
                }
            });
            preset.value = matched || '';
        }
    } else {
        status.textContent = userGeoLat !== null ? 'Using real GPS when available.' : 'No GPS yet — use header Location or a test override.';
        if (manual && !manual.matches(':focus')) {
            manual.value = '';
        }
        if (preset) preset.value = '';
    }
}

function updateLocationButtonUI() {
    const btn = document.getElementById('shareLocationButton');
    const label = document.getElementById('shareLocationLabel');
    if (!btn || !label) return;
    if (testLocationOverride) {
        btn.setAttribute('aria-pressed', 'true');
        label.textContent = 'Test loc';
        btn.classList.add('header-action-btn--location-on');
        btn.title = `Test override active: ${testLocationOverride.lat.toFixed(4)}, ${testLocationOverride.lon.toFixed(4)} — click to use real GPS instead`;
        return;
    }
    const ok = userGeoLat !== null && userGeoLon !== null;
    btn.setAttribute('aria-pressed', ok ? 'true' : 'false');
    label.textContent = ok ? 'Location on' : 'Location';
    btn.classList.toggle('header-action-btn--location-on', ok);
    btn.title =
        'Approximate location helps disambiguate places (e.g. same street name). Not sent as journey start/end unless you type it.';
}

function applyGeoPosition(pos) {
    userGeoLat = pos.coords.latitude;
    userGeoLon = pos.coords.longitude;
    updateLocationButtonUI();
    if (!testLocationOverride) syncTestLocationPanel();
}

/**
 * Ask the browser for the user's position. Browsers only show a clear permission
 * prompt when this runs after a user gesture — use the header "Location" button
 * for that. A silent attempt on load may succeed or fail without any in-app text.
 */
function requestShareLocation(isUserClick) {
    if (isUserClick) {
        clearTestLocationOverride();
    }
    if (!navigator.geolocation) {
        if (isUserClick) showShareFeedback('Geolocation not supported in this browser');
        return;
    }
    navigator.geolocation.getCurrentPosition(
        (pos) => {
            applyGeoPosition(pos);
            if (isUserClick) {
                showShareFeedback('Location saved — helps disambiguate place names');
            } else {
                // eslint-disable-next-line no-console
                console.log('[Geo] User location:', userGeoLat, userGeoLon);
            }
        },
        (err) => {
            updateLocationButtonUI();
            if (isUserClick) {
                const msg =
                    err && err.code === 1
                        ? 'Location blocked — allow it in the browser address bar'
                        : 'Could not read location';
                showShareFeedback(msg);
            } else {
                // eslint-disable-next-line no-console
                console.log('[Geo] Geolocation unavailable:', err && err.message ? err.message : err);
            }
        },
        { enableHighAccuracy: false, timeout: 15000, maximumAge: 300000 }
    );
}

testLocationOverride = loadTestLocationFromStorage();
syncTestLocationPanel();
updateLocationButtonUI();

if (navigator.geolocation) {
    requestShareLocation(false);
    navigator.geolocation.watchPosition(
        (pos) => applyGeoPosition(pos),
        () => {},
        { enableHighAccuracy: false, timeout: 15000, maximumAge: 60000 }
    );
}

const shareLocationButton = document.getElementById('shareLocationButton');
if (shareLocationButton) {
    shareLocationButton.addEventListener('click', () => requestShareLocation(true));
}

(function initTestLocationPanel() {
    const applyBtn = document.getElementById('testLocationApplyBtn');
    const clearBtn = document.getElementById('testLocationClearBtn');
    const preset = document.getElementById('testLocationPreset');
    const manual = document.getElementById('testLocationManual');
    if (!applyBtn || !clearBtn) return;

    applyBtn.addEventListener('click', () => {
        let loc = null;
        if (manual && manual.value.trim()) {
            loc = parseLatLonPair(manual.value);
            if (!loc) {
                showShareFeedback('Invalid lat, lon — use two numbers');
                return;
            }
        } else if (preset && preset.value) {
            loc = parseLatLonPair(preset.value.replace(',', ' '));
        }
        if (!loc) {
            showShareFeedback('Choose a preset or enter lat, lon');
            return;
        }
        setTestLocationOverride(loc);
        showShareFeedback('Test location applied');
    });

    clearBtn.addEventListener('click', () => {
        clearTestLocationOverride();
        showShareFeedback('Test location cleared');
    });
})();

/** Get journey history from localStorage. */
function getJourneyHistory() {
    try {
        const raw = localStorage.getItem(JOURNEY_HISTORY_KEY);
        return raw ? JSON.parse(raw) : {};
    } catch (e) {
        return {};
    }
}

/** Save updated journey history to localStorage. */
function saveJourneyHistory(history) {
    try {
        localStorage.setItem(JOURNEY_HISTORY_KEY, JSON.stringify(history));
    } catch (e) {
        console.warn('[JourneyHistory] Failed to save:', e);
    }
}

/**
 * Build the extra payload fields for journey disambiguation.
 * Included in every /chat request so the backend can use them for scoring.
 */
function getJourneyContextPayload() {
    const payload = {};
    if (testLocationOverride) {
        payload.userLat = testLocationOverride.lat;
        payload.userLon = testLocationOverride.lon;
    } else if (userGeoLat !== null && userGeoLon !== null) {
        payload.userLat = userGeoLat;
        payload.userLon = userGeoLon;
    }
    const history = getJourneyHistory();
    if (history && Object.keys(history).length > 0) {
        payload.journeyHistory = history;
    }
    return payload;
}

// Journey planner UI elements (interface disabled)
const journeyInputsSection = null;
const journeyPlannerToggle = null;
const fromInput = null;
const toInput = null;
const dateInput = null;
const timeInput = null;
const planBtn = null;

// Text shortcut UI elements
const shortcutKeyInput = document.getElementById('shortcut-key-input');
const shortcutValueInput = document.getElementById('shortcut-value-input');
const shortcutSaveBtn = document.getElementById('shortcut-save-btn');
const shortcutList = document.getElementById('shortcut-list');

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
let showStarredOnly = false;
let selectedChatIds = new Set();

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
    selectedChatIds.clear();
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
    selectedChatIds.delete(chatId);
    if (activeChatId === chatId) {
        activeChatId = conversations[0]?.id || null;
        if (activeChatId) setActiveChat(activeChatId);
        else if (chatMessages) clearMessages();
    }
    saveConversations();
    renderChatList(currentSearchTerm);
    updateDeleteSelectedBar();
}

function deleteSelectedChats() {
    if (selectedChatIds.size === 0) return;
    const wasActive = activeChatId && selectedChatIds.has(activeChatId);
    conversations = conversations.filter((c) => !selectedChatIds.has(c.id));
    selectedChatIds.clear();
    if (wasActive) {
        activeChatId = conversations[0]?.id || null;
        if (activeChatId) setActiveChat(activeChatId);
        else if (chatMessages) clearMessages();
    }
    saveConversations();
    renderChatList(currentSearchTerm);
    updateDeleteSelectedBar();
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
    updateDeleteSelectedBar();
}

function getConversationTranscript(chatId) {
    const convo = conversations.find((c) => c.id === chatId);
    if (!convo || !convo.messages || convo.messages.length === 0) {
        return null;
    }
    const lines = ['Travel assistant – Chat: ' + (convo.title || 'Untitled'), '', '---', ''];
    convo.messages.forEach((m) => {
        const label = m.sender === 'user' ? 'You' : 'Assistant';
        const text = (m.text || '').replace(/\n/g, '\n  ');
        lines.push(label + ': ' + text);
        lines.push('');
    });
    return lines.join('\n').trim();
}

function shareConversation(chatId) {
    const text = getConversationTranscript(chatId || activeChatId);
    if (!text) {
        if (typeof alert !== 'undefined') alert('Nothing to share in this conversation yet.');
        return;
    }
    if (typeof navigator.share === 'function') {
        navigator.share({
            title: 'Travel assistant conversation',
            text: text,
        }).then(() => {
            showShareFeedback('Shared');
        }).catch((err) => {
            if (err.name !== 'AbortError') copyToClipboardAndFeedback(text);
        });
    } else {
        copyToClipboardAndFeedback(text);
    }
}

function copyToClipboardAndFeedback(text) {
    if (!navigator.clipboard || !navigator.clipboard.writeText) {
        if (typeof alert !== 'undefined') alert('Copy not supported. Transcript:\n\n' + text.slice(0, 500) + (text.length > 500 ? '…' : ''));
        return;
    }
    navigator.clipboard.writeText(text).then(() => {
        showShareFeedback('Copied to clipboard');
    }).catch(() => {
        if (typeof alert !== 'undefined') alert('Could not copy. Try selecting the text manually.');
    });
}

function showShareFeedback(message) {
    const el = document.getElementById('shareFeedbackToast');
    if (el) {
        el.textContent = message;
        el.classList.remove('share-feedback-hidden');
        clearTimeout(el._hideTimer);
        el._hideTimer = setTimeout(() => {
            el.classList.add('share-feedback-hidden');
        }, 2000);
    }
}

function updateDeleteSelectedBar() {
    const bar = document.getElementById('chatListDeleteSelectedBar');
    if (!bar) return;
    bar.classList.toggle('hidden', selectedChatIds.size === 0);
    const countEl = bar.querySelector('.chat-list-delete-selected-count');
    if (countEl) countEl.textContent = selectedChatIds.size;
}

function getWelcomeMessageHtml() {
    const greeting = currentUser
        ? `Hi ${escapeHtml(currentUser)}, I'm your travel assistant. I can help you with:`
        : "Hi, I'm your travel assistant. I can help you with:";
    return `
        <p>${greeting}</p>
        <ul>
            <li>Bus and train timetables (next arrivals)</li>
            <li>Live service status and disruption updates (TfL lines and bus routes)</li>
            <li>Journey planning in London with step-by-step route legs</li>
        </ul>
        <p>You can ask things like: "What's the bus schedule for Blackbird Hill?", "Is the Jubilee line running?", or "Plan a journey from Neasden to Waterloo".</p>
        <p class="welcome-location-hint">Optional: use the <strong>Location</strong> button in the header to share approximate position — it helps when several places share the same name. The app does not use it as your journey start unless you type that.</p>
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
    if (showStarredOnly) filtered = filtered.filter((c) => c.starred);
    const sorted = [...filtered].sort((a, b) => {
        if (a.starred !== b.starred) return a.starred ? -1 : 1;
        return (b.updatedAt || 0) - (a.updatedAt || 0);
    });

    chatList.innerHTML = '';
    if (showStarredOnly && sorted.length === 0) {
        const empty = document.createElement('li');
        empty.className = 'chat-list-empty';
        empty.textContent = 'No starred chats';
        chatList.appendChild(empty);
    }
    sorted.forEach((convo) => {
        const li = document.createElement('li');
        li.className = 'chat-list-item';
        li.dataset.chatId = convo.id;
        if (selectedChatIds.has(convo.id)) li.classList.add('chat-list-item-selected');

        const checkbox = document.createElement('button');
        checkbox.type = 'button';
        checkbox.className = 'chat-list-item-checkbox';
        checkbox.setAttribute('aria-label', selectedChatIds.has(convo.id) ? 'Deselect' : 'Select');
        checkbox.innerHTML = selectedChatIds.has(convo.id) ? '✓' : '';
        checkbox.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleChatSelected(convo.id);
        });

        const starBtn = document.createElement('button');
        starBtn.type = 'button';
        starBtn.className = 'chat-list-item-star';
        starBtn.setAttribute('aria-label', convo.starred ? 'Unstar' : 'Star');
        starBtn.textContent = convo.starred ? '★' : '☆';
        starBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            toggleStar(convo.id);
        });

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
        const shareMenuBtn = document.createElement('button');
        shareMenuBtn.type = 'button';
        shareMenuBtn.className = 'chat-list-item-menu-item';
        shareMenuBtn.textContent = 'Share';
        shareMenuBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            menu.classList.add('hidden');
            shareConversation(convo.id);
        });
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
        menu.appendChild(shareMenuBtn);
        menu.appendChild(renameBtn);
        menu.appendChild(deleteBtn);

        li.appendChild(checkbox);
        li.appendChild(starBtn);
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

const chatListDeleteSelectedBar = document.getElementById('chatListDeleteSelectedBar');
if (chatListDeleteSelectedBar) {
    const deleteSelectedBtn = chatListDeleteSelectedBar.querySelector('.chat-list-delete-selected-btn');
    if (deleteSelectedBtn) {
        deleteSelectedBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteSelectedChats();
        });
    }
}

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

// -------- Voice input (speech-to-text) --------
const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isListening = false;

if (SpeechRecognitionAPI) {
    recognition = new SpeechRecognitionAPI();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-GB';

    recognition.onresult = (event) => {
        let finalTranscript = '';
        let interimTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
        }
        if (finalTranscript && userInput) {
            const current = (userInput.value || '').trim();
            userInput.value = current ? current + ' ' + finalTranscript : finalTranscript;
        }
    };

    recognition.onend = () => {
        isListening = false;
        if (voiceInputButton) {
            voiceInputButton.classList.remove('voice-btn-listening');
            voiceInputButton.setAttribute('aria-label', 'Voice input (speak to type)');
        }
    };

    recognition.onerror = (event) => {
        isListening = false;
        if (voiceInputButton) voiceInputButton.classList.remove('voice-btn-listening');
        if (event.error === 'not-allowed') {
            console.warn('Voice input: permission denied or blocked.');
        }
    };
}

if (voiceInputButton) {
    voiceInputButton.addEventListener('click', () => {
        if (!recognition) {
            console.warn('Speech recognition not supported in this browser.');
            return;
        }
        if (isListening) {
            recognition.stop();
            return;
        }
        isListening = true;
        voiceInputButton.classList.add('voice-btn-listening');
        voiceInputButton.setAttribute('aria-label', 'Listening… Click to stop');
        recognition.start();
    });
}

// Load voices for TTS (some browsers need this after user interaction)
if (window.speechSynthesis) {
    speechSynthesis.getVoices();
    window.addEventListener('voiceschanged', () => speechSynthesis.getVoices());
}

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

    // Send message to backend (include geolocation + journey history)
    const chatPayload = { message: message, ...getJourneyContextPayload() };
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(chatPayload)
    })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator
            removeTypingIndicator(typingId);
            // Persist updated journey history if backend sent it back
            if (data.updated_journey_history) {
                saveJourneyHistory(data.updated_journey_history);
            }
            handleChatResponse(data);
            // For testing: hard-coded demo of live traffic map for Wembley.
            maybeRenderTrafficTestMap(message);
        })
        .catch(error => {
            removeTypingIndicator(typingId);
            addMessageAndStore('Sorry, I encountered an error. Please try again.', 'bot');
            console.error('Error:', error);
        });
}

/**
 * Send a short follow-up reply back to the chatbot without using the text box.
 * Used for clicks on disambiguation maps so that picking a marker behaves like
 * replying with "1", "2", or the place name in the normal chat flow.
 */
function sendDisambiguationChoice(choiceText) {
    const message = (choiceText || '').trim();
    if (!message) return;

    // Show the user's selection in the conversation.
    addMessageAndStore(message, 'user');

    // Show typing indicator
    const typingId = showTypingIndicator();

    // Send message to backend (include geolocation + journey history)
    const chatPayload = { message, ...getJourneyContextPayload() };
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(chatPayload)
    })
        .then((response) => response.json())
        .then((data) => {
            removeTypingIndicator(typingId);
            if (data.updated_journey_history) {
                saveJourneyHistory(data.updated_journey_history);
            }
            handleChatResponse(data);
        })
        .catch((error) => {
            removeTypingIndicator(typingId);
            addMessageAndStore('Sorry, I encountered an error. Please try again.', 'bot');
            // eslint-disable-next-line no-console
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
        const actionsWrap = document.createElement('div');
        actionsWrap.className = 'message-actions-wrap';
        const readAloudBtn = document.createElement('button');
        readAloudBtn.type = 'button';
        readAloudBtn.className = 'message-action-btn';
        readAloudBtn.setAttribute('aria-label', 'Read aloud');
        readAloudBtn.innerHTML = '<span class="message-action-icon" aria-hidden="true">🔊</span><span class="message-action-label">Read aloud</span>';
        readAloudBtn.addEventListener('click', () => speakMessageText(text));
        const shareBtn = document.createElement('button');
        shareBtn.type = 'button';
        shareBtn.className = 'message-action-btn';
        shareBtn.setAttribute('aria-label', 'Share this message');
        shareBtn.innerHTML = '<span class="message-action-icon" aria-hidden="true">📤</span><span class="message-action-label">Share</span>';
        shareBtn.addEventListener('click', () => shareMessageText(text));
        actionsWrap.appendChild(readAloudBtn);
        actionsWrap.appendChild(shareBtn);
        contentDiv.appendChild(actionsWrap);
    }

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendReadAloudAndShareButtons(container, text) {
    if (!container) return;
    const useText = (text && String(text).trim()) ? String(text).trim() : 'No content to read or share.';
    const actionsWrap = document.createElement('div');
    actionsWrap.className = 'message-actions-wrap';
    if (container.classList && (container.classList.contains('timetable-card') || container.classList.contains('disruption-card'))) {
        actionsWrap.classList.add('message-actions-inside-card');
    }
    const readAloudBtn = document.createElement('button');
    readAloudBtn.type = 'button';
    readAloudBtn.className = 'message-action-btn';
    readAloudBtn.setAttribute('aria-label', 'Read aloud');
    readAloudBtn.innerHTML = '<span class="message-action-icon" aria-hidden="true">🔊</span><span class="message-action-label">Read aloud</span>';
    readAloudBtn.addEventListener('click', () => speakMessageText(useText));
    const shareBtn = document.createElement('button');
    shareBtn.type = 'button';
    shareBtn.className = 'message-action-btn';
    shareBtn.setAttribute('aria-label', 'Share this message');
    shareBtn.innerHTML = '<span class="message-action-icon" aria-hidden="true">📤</span><span class="message-action-label">Share</span>';
    shareBtn.addEventListener('click', () => shareMessageText(useText));
    actionsWrap.appendChild(readAloudBtn);
    actionsWrap.appendChild(shareBtn);
    container.appendChild(actionsWrap);
}

function speakMessageText(text) {
    if (!text || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 0.95;
    u.pitch = 1;
    const voices = speechSynthesis.getVoices();
    const en = voices.find((v) => v.lang.startsWith('en'));
    if (en) u.voice = en;
    window.speechSynthesis.speak(u);
}

function shareMessageText(text) {
    const trimmed = typeof text === 'string' ? text.trim() : '';
    if (!trimmed) {
        showShareFeedback('Nothing to share');
        return;
    }
    try {
        if (typeof navigator.share === 'function') {
            navigator.share({
                title: 'Travel assistant',
                text: trimmed,
            }).then(() => showShareFeedback('Shared')).catch((err) => {
                if (err.name !== 'AbortError') copyMessageToClipboard(trimmed);
            });
        } else {
            copyMessageToClipboard(trimmed);
        }
    } catch (e) {
        copyMessageToClipboard(trimmed);
    }
}

function copyMessageToClipboard(text) {
    function showDone(success) {
        showShareFeedback(success ? 'Copied to clipboard' : 'Could not copy');
    }
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
        navigator.clipboard.writeText(text).then(() => showDone(true)).catch(() => {
            fallbackCopyText(text, showDone);
        });
    } else {
        fallbackCopyText(text, showDone);
    }
}

function fallbackCopyText(text, callback) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    ta.style.top = '0';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    let ok = false;
    try {
        ok = document.execCommand('copy');
    } catch (e) {}
    document.body.removeChild(ta);
    callback(!!ok);
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

    // Update entities — render as styled tag chips grouped by category
    if (data.entities && Object.keys(data.entities).length > 0) {
        entitiesDisplay.innerHTML = '';

        // Category groupings and colors for entity keys
        const categoryMap = {
            origin:             { label: 'Origin',           cat: 'journey', color: '#2563eb' },
            destination:        { label: 'Destination',      cat: 'journey', color: '#2563eb' },
            via:                { label: 'Via',              cat: 'journey', color: '#2563eb' },
            date:               { label: 'Date',             cat: 'journey', color: '#2563eb' },
            time:               { label: 'Time',             cat: 'journey', color: '#2563eb' },
            time_preference:    { label: 'Time Pref',        cat: 'journey', color: '#2563eb' },
            mode:               { label: 'Modes',            cat: 'prefs',   color: '#7c3aed' },
            journey_preference: { label: 'Optimise',         cat: 'prefs',   color: '#7c3aed' },
            travel_mode:        { label: 'Mode',             cat: 'prefs',   color: '#7c3aed' },
            timetable_mode:     { label: 'Timetable',        cat: 'prefs',   color: '#7c3aed' },
            accessibility:      { label: 'Accessibility',    cat: 'prefs',   color: '#7c3aed' },
            bus_route:          { label: 'Bus Route',        cat: 'transport', color: '#dc2626' },
            line:               { label: 'Line',             cat: 'transport', color: '#dc2626' },
            route:              { label: 'Route',            cat: 'transport', color: '#dc2626' },
            location:           { label: 'Location',         cat: 'transport', color: '#059669' },
            road:               { label: 'Road',             cat: 'transport', color: '#059669' },
            llm_urgency:        { label: 'Urgency',          cat: 'context',  color: '#d97706' },
            llm_mood:           { label: 'Mood',             cat: 'context',  color: '#d97706' },
            llm_utterance_type: { label: 'Type',             cat: 'context',  color: '#d97706' },
        };

        // Filter out internal/private keys and technical IDs not useful to display
        const hiddenKeys = new Set(['_', 'from_id', 'to_id', 'timeIs', 'timetable_stop_source', 'csv_stop_name']);
        const entries = Object.entries(data.entities)
            .filter(([key]) => !key.startsWith('_') && !hiddenKeys.has(key));

        if (entries.length === 0) {
            entitiesDisplay.textContent = 'None detected';
            return;
        }

        const placeKeys = new Set(['origin', 'destination', 'via']);

        entries.forEach(([key, value]) => {
            const meta = categoryMap[key] || { label: key, cat: 'other', color: '#6b7280' };
            const chip = document.createElement('span');
            const isPlace = placeKeys.has(key);
            chip.className = 'entity-chip entity-cat-' + meta.cat + (isPlace ? ' entity-chip-place' : '');
            chip.style.cssText = `
                display: inline-block;
                margin: 2px 4px 2px 0;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 0.8rem;
                background: ${meta.color}18;
                color: ${meta.color};
                border: 1px solid ${meta.color}40;
                white-space: nowrap;
            `;
            const labelHtml = `<strong>${escapeHtml(meta.label)}:</strong>`;
            const valueHtml = isPlace
                ? ` <span class="entity-chip-value">${escapeHtml(String(value))}</span>`
                : ` ${escapeHtml(String(value))}`;
            chip.innerHTML = labelHtml + valueHtml;
            entitiesDisplay.appendChild(chip);
        });
    } else {
        entitiesDisplay.innerHTML = '';
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

/**
 * Build a journey card from the journey-planner summary format (journey_planner.py).
 * Expects: { departure, arrival, duration, legs: [{ mode, detail, duration, steps?, stops?, ... }], fare_pence? }
 */
function buildJourneyCard(journey) {
    const card = document.createElement('div');
    card.className = 'journey-card';

    const farePounds = journey.fare_pence != null ? (journey.fare_pence / 100).toFixed(2) : null;

    // Top section: Departure, Arrival, Duration (and optional Fare) – same structure as JourneySummary in journey_planner.py
    const top = document.createElement('div');
    top.className = 'journey-card-top';
    top.innerHTML = `
    <div class="journey-card-summary">
      <div class="journey-card-row">
        <span class="journey-card-label">Departure</span>
        <span class="journey-card-value">${escapeHtml(journey.departure || '—')}</span>
      </div>
      <div class="journey-card-row">
        <span class="journey-card-label">Arrival</span>
        <span class="journey-card-value">${escapeHtml(journey.arrival || '—')}</span>
      </div>
      <div class="journey-card-row">
        <span class="journey-card-label">Duration</span>
        <span class="journey-card-value"><strong>${journey.duration != null ? journey.duration : '—'}</strong> mins</span>
      </div>
      ${farePounds != null ? `<div class="journey-card-row"><span class="journey-card-label">Fare</span><span class="journey-card-value journey-card-fare">£${farePounds} (off peak)</span></div>` : ''}
    </div>
  `;
    card.appendChild(top);

    // Legs: each leg as mode + detail + duration (JourneyLeg format from journey_planner.py)
    const timeline = document.createElement('div');
    timeline.className = 'journey-card-timeline';
    const legsHeading = document.createElement('div');
    legsHeading.className = 'journey-card-legs-heading';
    legsHeading.textContent = 'Route';
    timeline.appendChild(legsHeading);

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
        const detailText = leg.detail ? `${leg.mode || 'Leg'} – ${leg.detail}` : (leg.mode || 'Leg');
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
        <div class="journey-card-leg-detail">${escapeHtml(detailText)}</div>
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
      <div class="journey-card-leg-detail">${escapeHtml(destLabel)}</div>
    </div>
  `;
    timeline.appendChild(destRow);
    card.appendChild(timeline);

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
                    } else {
                        // No inline steps and no external maps allowed; show a gentle note instead.
                        expandable.innerHTML = '<div class="journey-leg-note">Directions are not available to open in an external map here.</div>';
                        expandable.classList.remove('hidden');
                        link.textContent = 'Hide directions';
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

    appendReadAloudAndShareButtons(card, messageText);
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

    appendReadAloudAndShareButtons(card, messageText);
    wrap.appendChild(card);
    chatMessages.appendChild(wrap);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function renderTimetableDisambiguationMap(disamb) {
    if (!disamb || !Array.isArray(disamb.options) || disamb.options.length === 0) return;
    if (!window.google || !google.maps) return;

    const botMessages = chatMessages ? chatMessages.querySelectorAll('.message.bot-message') : null;
    if (!botMessages || botMessages.length === 0) return;
    const lastBot = botMessages[botMessages.length - 1];
    const content = lastBot.querySelector('.message-content');
    if (!content) return;

    // Remove any existing disambiguation map in this message
    const existing = content.querySelector('.timetable-disamb-map');
    if (existing) existing.remove();

    const mapWrap = document.createElement('div');
    mapWrap.className = 'timetable-disamb-map';
    content.appendChild(mapWrap);

    // Collect valid coordinates, if present on options (bus stops or train stations)
    const points = disamb.options
        .map((opt, index) => {
            const lat = typeof opt.lat === 'number' ? opt.lat : (opt.lat ? Number(opt.lat) : NaN);
            const lon = typeof opt.lon === 'number' ? opt.lon : (opt.lon ? Number(opt.lon) : NaN);
            if (!isFinite(lat) || !isFinite(lon)) return null;
            return {
                position: { lat, lng: lon },
                label: String(index + 1),
                title: opt.label || opt.name || '',
            };
        })
        .filter(Boolean);

    // Case 1: we have explicit coordinates (bus or multi-station train disambiguation)
    if (points.length) {
        const map = new google.maps.Map(mapWrap, {
            center: points[0].position,
            zoom: 15,
            mapTypeControl: false,
            streetViewControl: false,
            fullscreenControl: false,
            clickableIcons: false,
        });

        const bounds = new google.maps.LatLngBounds();

        // Score lookup for label coloring.
        // Journey planner passes scores in a separate {id: score} dict;
        // timetable disambiguation embeds score inside each option object.
        const scoresDict = disamb.scores || {};

        points.forEach((pt) => {
            // Color marker based on score (green = high, orange = medium, red = low)
            const opt = disamb.options[parseInt(pt.label) - 1] || {};
            const score = scoresDict[opt.id || ''] || opt.score || 0;
            let markerColor = '#EA4335'; // red (low)
            if (score >= 0.7) markerColor = '#34A853'; // green (high)
            else if (score >= 0.4) markerColor = '#FBBC05'; // yellow/orange (medium)

            const marker = new google.maps.Marker({
                position: pt.position,
                map,
                label: {
                    text: pt.label,
                    color: '#FFFFFF',
                    fontWeight: 'bold',
                },
                title: `${pt.title} (${Math.round(score * 100)}%)`,
                icon: {
                    path: google.maps.SymbolPath.CIRCLE,
                    fillColor: markerColor,
                    fillOpacity: 1,
                    strokeColor: '#FFFFFF',
                    strokeWeight: 2,
                    scale: 14,
                },
            });
            marker.addListener('click', () => {
                if (typeof sendDisambiguationChoice === 'function') {
                    sendDisambiguationChoice(pt.label);
                }
            });
            bounds.extend(pt.position);
        });

        if (!bounds.isEmpty()) {
            map.fitBounds(bounds);
        }
        return;
    }

    // Case 2: train platform/direction disambiguation with no per-option coords:
    // show a single marker for the station itself using geocoding of disamb.query.
    if (disamb.train_direction_disambiguation && disamb.query) {
        const geocoder = new google.maps.Geocoder();
        geocoder.geocode({ address: disamb.query }, (results, status) => {
            if (status !== 'OK' || !results || !results[0]) {
                content.removeChild(mapWrap);
                return;
            }
            const loc = results[0].geometry.location;
            const center = { lat: loc.lat(), lng: loc.lng() };
            const map = new google.maps.Map(mapWrap, {
                center,
                zoom: 16,
                mapTypeControl: false,
                streetViewControl: false,
                fullscreenControl: false,
                clickableIcons: false,
            });
            // Single marker for the station
            // (no label number needed; there is only one physical station here)
            new google.maps.Marker({
                position: center,
                map,
                title: disamb.query,
            });
        });
        return;
    }

    // No coordinates and no query to geocode; remove empty container.
    content.removeChild(mapWrap);
}

// Hard-coded demo: show a Google Maps traffic layer centred on Wembley
// when the user asks specifically for "traffic in wembley".
function maybeRenderTrafficTestMap(originalMessage) {
    if (!originalMessage) return;
    const text = String(originalMessage).toLowerCase().trim();
    if (text !== 'traffic in wembley') return;
    if (!chatMessages) return;
    if (!window.google || !google.maps) {
        // Google Maps script not ready; nothing to render.
        return;
    }

    const wrap = document.createElement('div');
    wrap.className = 'message bot-message';

    const content = document.createElement('div');
    content.className = 'message-content';

    const heading = document.createElement('p');
    heading.textContent = 'Here is live traffic in Wembley (test view).';
    content.appendChild(heading);

    const mapDiv = document.createElement('div');
    mapDiv.className = 'traffic-test-map';
    mapDiv.style.width = '100%';
    mapDiv.style.height = '260px';
    mapDiv.style.borderRadius = '8px';
    mapDiv.style.overflow = 'hidden';
    mapDiv.style.marginTop = '8px';
    content.appendChild(mapDiv);

    wrap.appendChild(content);
    chatMessages.appendChild(wrap);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    const wembleyCenter = { lat: 51.5560, lng: -0.2796 }; // Approximate Wembley Stadium area
    const map = new google.maps.Map(mapDiv, {
        center: wembleyCenter,
        zoom: 13,
        mapTypeControl: false,
        streetViewControl: false,
        fullscreenControl: false,
    });

    const trafficLayer = new google.maps.TrafficLayer();
    trafficLayer.setMap(map);
}

/**
 * renderConfirmPinsMap — shows a Google Maps with two draggable pins (green = origin,
 * red = destination) attached to the last bot message. The user can drag them to
 * fine-tune exact locations, then click "Plan Route" to confirm.
 */
function renderConfirmPinsMap(pinData) {
    if (!pinData || !pinData.from || !pinData.to) return;
    if (!window.google || !google.maps) return;

    const botMessages = chatMessages ? chatMessages.querySelectorAll('.message.bot-message') : null;
    if (!botMessages || botMessages.length === 0) return;
    const lastBot = botMessages[botMessages.length - 1];
    const content = lastBot.querySelector('.message-content');
    if (!content) return;

    // Remove any previous confirm-pins map in this message
    const existing = content.querySelector('.confirm-pins-map-wrap');
    if (existing) existing.remove();

    // Outer wrapper
    const wrap = document.createElement('div');
    wrap.className = 'confirm-pins-map-wrap';

    // Map container (will be initialised after browser layout pass)
    const mapDiv = document.createElement('div');
    mapDiv.className = 'confirm-pins-map';
    wrap.appendChild(mapDiv);

    // Legend row
    const legend = document.createElement('div');
    legend.className = 'confirm-pins-legend';
    legend.innerHTML = `
        <span class="confirm-pins-legend-item">
            <svg width="16" height="16" viewBox="0 0 16 16"><circle cx="8" cy="8" r="8" fill="#34A853"/><text x="8" y="12" text-anchor="middle" fill="white" font-size="9" font-weight="bold">A</text></svg>
            ${pinData.from.name || 'Origin'}
        </span>
        <span class="confirm-pins-legend-item">
            <svg width="16" height="16" viewBox="0 0 16 16"><circle cx="8" cy="8" r="8" fill="#EA4335"/><text x="8" y="12" text-anchor="middle" fill="white" font-size="9" font-weight="bold">B</text></svg>
            ${pinData.to.name || 'Destination'}
        </span>`;
    wrap.appendChild(legend);

    // "Plan Route" button
    const btn = document.createElement('button');
    btn.textContent = 'Plan Route';
    btn.className = 'confirm-pins-btn';

    // Keep track of current pin positions (mutable as user drags)
    const pinState = {
        from: { lat: Number(pinData.from.lat), lon: Number(pinData.from.lon) },
        to:   { lat: Number(pinData.to.lat),   lon: Number(pinData.to.lon)   },
    };

    btn.onclick = () => {
        btn.disabled = true;
        btn.textContent = 'Planning…';
        const payload = {
            message: '',
            confirmPins: {
                from: { lat: pinState.from.lat, lon: pinState.from.lon },
                to:   { lat: pinState.to.lat,   lon: pinState.to.lon   },
            },
        };
        if (testLocationOverride) {
            payload.userLat = testLocationOverride.lat;
            payload.userLon = testLocationOverride.lon;
        } else if (userGeoLat != null) {
            payload.userLat = userGeoLat;
            payload.userLon = userGeoLon;
        }
        fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        })
        .then(r => r.json())
        .then(d => {
            btn.textContent = 'Route planned ✓';
            handleChatResponse(d);
        })
        .catch(() => {
            btn.disabled = false;
            btn.textContent = 'Plan Route';
        });
    };
    wrap.appendChild(btn);

    // Insert the whole wrap BEFORE the Read-aloud/Share buttons so it sits
    // naturally between the message text and the action bar.
    const actionsWrap = content.querySelector('.message-actions-wrap');
    if (actionsWrap) {
        content.insertBefore(wrap, actionsWrap);
    } else {
        content.appendChild(wrap);
    }
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Initialise the Google Map AFTER the browser has done a layout pass so the
    // mapDiv has real pixel dimensions (Maps silently fails on zero-size divs).
    requestAnimationFrame(() => {
        setTimeout(() => {
            const fromPos = { lat: pinState.from.lat, lng: pinState.from.lon };
            const toPos   = { lat: pinState.to.lat,   lng: pinState.to.lon   };
            const center  = {
                lat: (fromPos.lat + toPos.lat) / 2,
                lng: (fromPos.lng + toPos.lng) / 2,
            };

            const map = new google.maps.Map(mapDiv, {
                center,
                zoom: 13,
                mapTypeControl: false,
                streetViewControl: false,
                fullscreenControl: false,
                clickableIcons: false,
                gestureHandling: 'cooperative',
            });

            // Fit both pins in view
            const bounds = new google.maps.LatLngBounds();
            bounds.extend(fromPos);
            bounds.extend(toPos);
            map.fitBounds(bounds, { top: 40, right: 40, bottom: 40, left: 40 });

            // Origin marker — green A, draggable
            const fromMarker = new google.maps.Marker({
                position: fromPos,
                map,
                draggable: true,
                title: pinData.from.name || 'Origin',
                label: { text: 'A', color: '#fff', fontWeight: 'bold', fontSize: '13px' },
                icon: {
                    path: google.maps.SymbolPath.CIRCLE,
                    fillColor: '#34A853',
                    fillOpacity: 1,
                    strokeColor: '#fff',
                    strokeWeight: 2,
                    scale: 16,
                },
                zIndex: 2,
            });

            // Destination marker — red B, draggable
            const toMarker = new google.maps.Marker({
                position: toPos,
                map,
                draggable: true,
                title: pinData.to.name || 'Destination',
                label: { text: 'B', color: '#fff', fontWeight: 'bold', fontSize: '13px' },
                icon: {
                    path: google.maps.SymbolPath.CIRCLE,
                    fillColor: '#EA4335',
                    fillOpacity: 1,
                    strokeColor: '#fff',
                    strokeWeight: 2,
                    scale: 16,
                },
                zIndex: 2,
            });

            // Info windows
            const fromInfo = new google.maps.InfoWindow({
                content: `<strong>${pinData.from.name || 'Origin'}</strong><br><small>Drag to adjust</small>`,
            });
            const toInfo = new google.maps.InfoWindow({
                content: `<strong>${pinData.to.name || 'Destination'}</strong><br><small>Drag to adjust</small>`,
            });

            fromMarker.addListener('click', () => { toInfo.close(); fromInfo.open(map, fromMarker); });
            toMarker.addListener('click',   () => { fromInfo.close(); toInfo.open(map, toMarker); });

            fromMarker.addListener('dragend', (e) => {
                pinState.from.lat = e.latLng.lat();
                pinState.from.lon = e.latLng.lng();
                fromInfo.close();
            });
            toMarker.addListener('dragend', (e) => {
                pinState.to.lat = e.latLng.lat();
                pinState.to.lon = e.latLng.lng();
                toInfo.close();
            });

            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 50);
    });
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
        // Store journey payload and render journey cards inline in the chat,
        // but do not show or toggle any dedicated journey planner interface.
        addMessageAndStore(data.response || '', 'bot', { journeys: data.journeys, journeyOptions });
        appendJourneyCards(data.journeys, data.response || '', journeyOptions);
    } else if (data.timetable || data.disruption) {
        const payload = {};
        if (data.timetable) payload.timetable = data.timetable;
        if (data.disruption) payload.disruption = data.disruption;
        storeMessage(data.response || '', 'bot', payload);
        if (data.timetable) appendTimetableCard(data.timetable, data.response);
        if (data.disruption) appendDisruptionCard(data.disruption, data.response);
    } else if (data.response) {
        addMessageAndStore(data.response, 'bot');
    }

    // When the backend is asking "Which direction for '<query>'?" for bus timetable,
    // show an embedded map with the candidate stops.
    if (data.timetable_disambiguation) {
        renderTimetableDisambiguationMap(data.timetable_disambiguation);
    }
    // When journey planner asks "Which place did you mean?" with options that have coordinates, show map.
    if (data.journey_disambiguation && data.journey_disambiguation.options && data.journey_disambiguation.options.length > 0) {
        renderTimetableDisambiguationMap(data.journey_disambiguation);
    }
    // Both locations confirmed — show draggable pin map before fetching the route.
    if (data.confirm_pins && data.confirm_pins.from && data.confirm_pins.to) {
        renderConfirmPinsMap(data.confirm_pins);
    }

    updateInfoPanel(data);
}

function sendPlannedJourney() {
    // Journey planner interface is disabled; do nothing.
    return;
}

function initPlacesAutocomplete() {
    // Journey planner inputs are disabled; no Places autocomplete wiring needed.
}

// Expose callback for Google Places script (if configured)
window.initPlacesAutocomplete = initPlacesAutocomplete;

// Journey planner controls are disabled; no event wiring.

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

function updateChatListFilterButtons() {
    if (chatFilterAll) {
        chatFilterAll.classList.toggle('active', !showStarredOnly);
        chatFilterAll.setAttribute('aria-pressed', String(!showStarredOnly));
    }
    if (chatFilterStarred) {
        chatFilterStarred.classList.toggle('active', showStarredOnly);
        chatFilterStarred.setAttribute('aria-pressed', String(showStarredOnly));
    }
}

if (chatFilterAll) {
    chatFilterAll.addEventListener('click', () => {
        showStarredOnly = false;
        updateChatListFilterButtons();
        renderChatList(currentSearchTerm);
    });
}
if (chatFilterStarred) {
    chatFilterStarred.addEventListener('click', () => {
        showStarredOnly = true;
        updateChatListFilterButtons();
        renderChatList(currentSearchTerm);
    });
}

function getAvatarStorageKey() {
    return currentUser ? AVATAR_STORAGE_KEY_PREFIX + currentUser : null;
}

function getStoredAvatar() {
    const key = getAvatarStorageKey();
    if (!key) return null;
    try {
        return localStorage.getItem(key);
    } catch (e) {
        return null;
    }
}

function setStoredAvatar(dataUrl) {
    const key = getAvatarStorageKey();
    if (!key) return;
    try {
        if (dataUrl) {
            localStorage.setItem(key, dataUrl);
        } else {
            localStorage.removeItem(key);
        }
    } catch (e) {
        console.warn('Could not store avatar', e);
    }
}

function updateAvatarDisplay() {
    if (!accountMenuAvatar) return;
    const avatar = currentUser ? getStoredAvatar() : null;
    accountMenuAvatar.innerHTML = '';
    accountMenuAvatar.classList.remove('account-menu-avatar-img-wrap');
    if (avatar) {
        const img = document.createElement('img');
        img.src = avatar;
        img.alt = 'Profile';
        img.className = 'account-menu-avatar-img';
        accountMenuAvatar.appendChild(img);
        accountMenuAvatar.classList.add('account-menu-avatar-img-wrap');
    } else {
        accountMenuAvatar.textContent = '👤';
    }
    if (removeAvatarButton) {
        removeAvatarButton.classList.toggle('hidden', !avatar);
    }
}

function setLoggedIn(username) {
    currentUser = username || null;
    if (accountLoggedOut && accountLoggedIn && currentUsernameEl) {
        accountLoggedOut.classList.add('hidden');
        accountLoggedIn.classList.remove('hidden');
        currentUsernameEl.textContent = currentUser || '';
    }
    updateAvatarDisplay();
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
    updateAvatarDisplay();
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

if (changeAvatarButton && avatarFileInput) {
    changeAvatarButton.addEventListener('click', () => {
        avatarFileInput.value = '';
        avatarFileInput.click();
    });
}

if (avatarFileInput) {
    avatarFileInput.addEventListener('change', () => {
        const file = avatarFileInput.files && avatarFileInput.files[0];
        if (!file || !file.type.startsWith('image/')) return;
        const reader = new FileReader();
        reader.onload = () => {
            const dataUrl = reader.result;
            if (typeof dataUrl === 'string' && dataUrl.length < 500000) {
                setStoredAvatar(dataUrl);
                updateAvatarDisplay();
            }
        };
        reader.readAsDataURL(file);
    });
}

if (removeAvatarButton) {
    removeAvatarButton.addEventListener('click', () => {
        setStoredAvatar(null);
        updateAvatarDisplay();
    });
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
            // Once we know who we are, load any saved shortcuts.
            loadShortcuts();
        })
        .catch(() => {
            setLoggedOut();
            updateWelcomeMessage();
            loadShortcuts();
        });
});

// -------- Text shortcuts (user-defined words like "home", "uni") -----------

function renderShortcuts(shortcuts) {
    if (!shortcutList) return;
    shortcutList.innerHTML = '';
    if (!Array.isArray(shortcuts) || shortcuts.length === 0) {
        const li = document.createElement('li');
        li.className = 'shortcut-list-empty';
        li.textContent = currentUser
            ? 'No words yet. Add one above (Word + Means, then Save word).'
            : 'Sign in to add and see your words here.';
        shortcutList.appendChild(li);
        return;
    }

    shortcuts.forEach((item) => {
        const li = document.createElement('li');
        li.className = 'shortcut-list-item';
        const key = item.key || '';
        const value = item.value || '';
        li.innerHTML = `
            <span class="shortcut-key">${escapeHtml(key)}</span>
            <span class="shortcut-arrow">→</span>
            <span class="shortcut-value">${escapeHtml(value)}</span>
            <button type="button" class="shortcut-delete-btn" aria-label="Remove shortcut for ${escapeHtml(key)}">✕</button>
        `;
        const deleteBtn = li.querySelector('.shortcut-delete-btn');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', () => {
                deleteShortcut(key);
            });
        }
        shortcutList.appendChild(li);
    });
}

function loadShortcuts() {
    fetch('/shortcuts')
        .then((res) => res.json())
        .then((data) => {
            renderShortcuts(data);
        })
        .catch(() => {
            // Ignore errors; shortcuts are optional sugar.
        });
}

function saveShortcut() {
    if (!shortcutKeyInput || !shortcutValueInput) return;
    const key = (shortcutKeyInput.value || '').trim();
    const value = (shortcutValueInput.value || '').trim();
    if (!key || !value) {
        // eslint-disable-next-line no-alert
        alert('Please fill in both the word and what it should mean.');
        return;
    }

    fetch('/shortcuts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key, value }),
    })
        .then((res) => res.json())
        .then((data) => {
            if (data.error) {
                // eslint-disable-next-line no-alert
                alert(data.error);
                return;
            }
            shortcutKeyInput.value = '';
            shortcutValueInput.value = '';
            loadShortcuts();
        })
        .catch(() => {
            // eslint-disable-next-line no-alert
            alert('Could not save shortcut. Please try again.');
        });
}

function deleteShortcut(key) {
    if (!key) return;
    fetch(`/shortcuts/${encodeURIComponent(key)}`, {
        method: 'DELETE',
    })
        .then((res) => res.json())
        .then(() => {
            loadShortcuts();
        })
        .catch(() => {
            // eslint-disable-next-line no-alert
            alert('Could not remove that shortcut. Please try again.');
        });
}

if (shortcutSaveBtn) {
    shortcutSaveBtn.addEventListener('click', saveShortcut);
}
