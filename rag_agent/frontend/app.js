const chatHistory = document.getElementById('chatHistory');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');

function addMessage(text, isUser = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${isUser ? 'user-msg' : 'system-msg'}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'msg-content';
    
    // Simple way to preserve line breaks from generation
    contentDiv.innerHTML = text.replace(/\\n/g, '<br>').replace(/\n/g, '<br>');
    
    msgDiv.appendChild(contentDiv);
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return contentDiv;
}

function showLoading() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message system-msg';
    msgDiv.id = 'loadingMsg';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'msg-content';
    contentDiv.innerHTML = `
        <div class="loading-dots">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    `;
    
    msgDiv.appendChild(contentDiv);
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function hideLoading() {
    const loadingMsg = document.getElementById('loadingMsg');
    if (loadingMsg) {
        loadingMsg.remove();
    }
}

async function handleSend() {
    const text = userInput.value.trim();
    if (!text) return;
    
    // UI Update
    userInput.value = '';
    userInput.disabled = true;
    sendBtn.disabled = true;
    
    // Add user message
    addMessage(text, true);
    
    // Show typing indicator
    showLoading();
    
    try {
        // Send request
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: text })
        });
        
        hideLoading();
        
        const data = await response.json();
        
        if (data.error) {
            addMessage(`Error: ${data.error}`);
        } else {
            addMessage(data.response);
        }
    } catch (error) {
        hideLoading();
        addMessage(`Network error: ${error.message}`);
    } finally {
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
}

sendBtn.addEventListener('click', handleSend);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleSend();
});
