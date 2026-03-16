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

// Journey planner UI elements
const journeyInputsSection = document.getElementById('journey-inputs');
const journeyPlannerToggle = document.getElementById('journey-planner-toggle');
const fromInput = document.getElementById('from-input');
const toInput = document.getElementById('to-input');
const dateInput = document.getElementById('date-input');
const timeInput = document.getElementById('time-input');
const planBtn = document.getElementById('plan-btn');

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
            <li>Current traffic and congestion on your route</li>
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

        const btn = e.target.closest('.journey-card-btn');
        if (!btn) return;
        // Map and external journey views are disabled to keep everything inside the app.
        return;
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
            clickableIcons: false, // prevent default POI popups with "View on Google Maps"
        });

        const bounds = new google.maps.LatLngBounds();
        points.forEach((pt) => {
            const marker = new google.maps.Marker({
                position: pt.position,
                map,
                label: pt.label,
                title: pt.title,
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

    // When the backend is asking "Which direction for '<query>'?" for bus timetable,
    // show an embedded map with the candidate stops.
    if (data.timetable_disambiguation) {
        renderTimetableDisambiguationMap(data.timetable_disambiguation);
    }
    // When journey planner asks "Which place did you mean?" with options that have coordinates, show map.
    if (data.journey_disambiguation && data.journey_disambiguation.options && data.journey_disambiguation.options.length > 0) {
        renderTimetableDisambiguationMap(data.journey_disambiguation);
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

function initPlacesAutocomplete() {
    if (!window.google || !google.maps || !google.maps.places) {
        return;
    }

    // Restrict Google Places autocomplete to Greater London area
    const londonBounds = new google.maps.LatLngBounds(
        // Approximate SW and NE corners of Greater London
        new google.maps.LatLng(51.28, -0.489),
        new google.maps.LatLng(51.686, 0.236)
    );

    if (fromInput) {
        const fromAutocomplete = new google.maps.places.Autocomplete(fromInput, {
            fields: ['geometry', 'name'],
            bounds: londonBounds,
            strictBounds: true,
        });
        fromAutocomplete.addListener('place_changed', () => {
            const place = fromAutocomplete.getPlace();
            if (
                place &&
                place.geometry &&
                place.geometry.location &&
                londonBounds.contains(place.geometry.location)
            ) {
                const lat = place.geometry.location.lat();
                const lng = place.geometry.location.lng();
                fromCoord = `${lat},${lng}`;
            } else {
                fromCoord = '';
            }
        });
    }

    if (toInput) {
        const toAutocomplete = new google.maps.places.Autocomplete(toInput, {
            fields: ['geometry', 'name'],
            bounds: londonBounds,
            strictBounds: true,
        });
        toAutocomplete.addListener('place_changed', () => {
            const place = toAutocomplete.getPlace();
            if (
                place &&
                place.geometry &&
                place.geometry.location &&
                londonBounds.contains(place.geometry.location)
            ) {
                const lat = place.geometry.location.lat();
                const lng = place.geometry.location.lng();
                toCoord = `${lat},${lng}`;
            } else {
                toCoord = '';
            }
        });
    }
}

// Expose callback for Google Places script (if configured)
window.initPlacesAutocomplete = initPlacesAutocomplete;

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
