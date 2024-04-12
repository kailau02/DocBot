const conversation = document.getElementById('conversation');
const botName = localStorage.getItem('botName');
let waitingForResponse = false;

const addConversationMessage = (sender, content, saveLine) => {
    const line = document.createElement("div");
    line.style.marginBottom = '20px';

    const senderPortion = document.createElement("b");
    senderPortion.textContent = sender.trim() + ": ";
    line.appendChild(senderPortion);

    const textPortion = document.createElement("span");
    textPortion.textContent = content.trim();
    line.appendChild(textPortion);

    conversation.appendChild(line);

    const history = JSON.parse(localStorage.getItem('history'));
    history.push({
        "sender": sender,
        "content": content
    });
    if (saveLine) {
        localStorage.setItem("history", JSON.stringify(history));
    }
}

const loadPage = () => {
    document.getElementById('bot-name').textContent = botName;

    const history = JSON.parse(localStorage.getItem('history'));
    history.forEach(obj => {
        addConversationMessage(obj.sender, obj.content, false)
    });
}


const setLoadingAnim = (isLoading) => {
    if (isLoading) {
        document.getElementById('send-btn').style.display = "none";
        document.getElementById('send-spinner').style.display = "";
    } else {
        document.getElementById('send-btn').style.display = "";
        document.getElementById('send-spinner').style.display = "none";
    }
}


const submitPrompt = async () => {
    if (waitingForResponse) {
        return;
    }
    const input = document.getElementById('input-prompt').value;
    if (input === '') {
        document.getElementById('error-text').textContent = "No input given.";
        return;
    }

    waitingForResponse = true;
    setLoadingAnim(true);

    document.getElementById('input-prompt').value = '';

    addConversationMessage("User", input, true);

    reqObj = {
        "data": input
    };

    const promptURL = 'http://127.0.0.1:8000/send_prompt';
    await fetch(promptURL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(reqObj)
    })
        .then(response => {
            if (response.status >= 300) {
                const errorMsg = "Error sending prompt";
                document.getElementById('error-text').textContent = errorMsg;
                waitingForResponse = false;
                setLoadingAnim(false);
                throw new Error(errorMsg);
            } else {
                document.getElementById('error-text').textContent = '';
            }
            return response.json();
        })
        .then(data => {
            const response = data.message;
            addConversationMessage(botName, response, true);
            waitingForResponse = false;
            setLoadingAnim(false);

        })
        .catch(error => {
            console.log("error", error);
            waitingForResponse = false;
            setLoadingAnim(false);

        })
}

const newChat = () => {
    window.location.href = "index.html";
}

loadPage();