let b64 = localStorage.getItem('b64');
let waitingForResponse = false;

const convertToBase64 = () => {
    //Read File
    var selectedFile = document.getElementById("inputFile").files;
    //Check File is not Empty
    if (selectedFile.length > 0) {
        // Select the very first file from list
        var fileToLoad = selectedFile[0];
        // FileReader function for read the file.
        var fileReader = new FileReader();
        var base64;
        // Onload of file read the file content
        fileReader.onload = function(fileLoadedEvent) {
            base64 = fileLoadedEvent.target.result;
            // Print data in console
            b64 = base64.split(/,(.*)/s)[1];
            localStorage.setItem('b64', b64);
        };
        // Convert data to base64
        fileReader.readAsDataURL(fileToLoad);
    }
}

const setLoadingAnim = (isLoading) => {
    if (isLoading) {
        document.getElementById('start-btn').style.display = "none";
        document.getElementById('start-spinner').style.display = "";
    } else {
        document.getElementById('start-btn').style.display = "";
        document.getElementById('start-spinner').style.display = "none";
    }
}

const submitPDF = async (e) => {
    if (waitingForResponse) {
        return;
    }
    if (document.getElementById('botNameField').value === '') {
        document.getElementById('error-text').textContent = 'Please enter a bot name.';
        return;
    }
    if (document.getElementById('inputFile').value === '') {
        document.getElementById('error-text').textContent = 'Please upload a PDF document.';
        return;
    }
    waitingForResponse = true;
    setLoadingAnim(true);
    e.preventDefault();
    const botName = document.getElementById('botNameField').value;
    localStorage.setItem("botName", botName);
    
    reqObj = {
        "data": b64
    };

    // const pdfURL = 'http://127.0.0.1:8000/pdf_test';
    const pdfURL = 'http://127.0.0.1:8000/begin_chat';
    await fetch(pdfURL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(reqObj)
    })
        .then(response => {
            if (response.status >= 300) {
                const errorMsg = "Error uploading document";
                document.getElementById('error-text').textContent = errorMsg;
                waitingForResponse = false;
                setLoadingAnim(false);
                throw new Error(errorMsg);
            }
            return response.json();
        })
        .then(data => {
            try {
                const history = [{
                    "sender": "Summary",
                    "content": data.message
                }];
                localStorage.setItem("history", JSON.stringify(history));
                window.location.href = "chat.html";
            } catch (error) {
                console.log('error');
            }
            waitingForResponse = false;
            setLoadingAnim(false);
        })
        .catch(error => {
            console.log("error", error);
            waitingForResponse = false;
            setLoadingAnim(false);
        })
}
