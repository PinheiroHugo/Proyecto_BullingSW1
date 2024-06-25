let rec;
let mediaRecorder;
let audioChunks = [];
let isProcessing = false;

if (!("webkitSpeechRecognition" in window)) {
    alert("Disculpa, no puedes usar la API");
} else {
    rec = new webkitSpeechRecognition();
    rec.lang = "es-AR";
    rec.continuous = true;
    rec.interimResults = true;
    rec.addEventListener("result", iniciar);

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

            mediaRecorder.ondataavailable = event => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                if (audioChunks.length > 0 && !isProcessing) {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    audioChunks = [];
                    if (mediaRecorder.state === "inactive") {
                        mediaRecorder.start(1000);
                    }
                }
            };

            if (mediaRecorder.state === "inactive") {
                mediaRecorder.start(1000);
            }
        })
        .catch(error => {
            console.error('Error al acceder al micrófono:', error);
        });
}

function iniciar(event) {
    if (isProcessing) return;

    console.log('leyendo');
    for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        document.getElementById('texto').innerHTML = transcript;

        const malasPalabras = [
            "mala",
            "palabra",
            "mala palabra",
            "ejemplo",
            "feo",
            "pendejo",
            "hijo de puta",
            "perra",
        ];

        for (const palabra of malasPalabras) {
            if (transcript.toLowerCase().includes(palabra.toLowerCase())) {
                isProcessing = true;
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                console.log('iniciar');
                enviarMalaPalabra(transcript, audioBlob);
                if (mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                }
                break;
            }
        }
    }
}

function enviarMalaPalabra(transcript, audioBlob) {
    const formData = new FormData();
    formData.append('mensaje', transcript);
    formData.append('fecha', '2023/11/05');
    formData.append('descripcion', 'Audio de posible acoso');
    formData.append('camara_id', '1');
    formData.append('es_queja', '0');
    formData.append('tipo', 'Audio');
    formData.append('file', audioBlob, 'audio.wav');

    /* const url = 'http://127.0.0.1:8000/api/evento/evidencia'; */
    const url = 'http://192.241.150.219/api/evento/evidencia';

    fetch(url, {
        method: 'POST',
        body: formData,
    })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            isProcessing = false;
            if (mediaRecorder.state === "inactive") {
                mediaRecorder.start(1000);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            isProcessing = false;
            if (mediaRecorder.state === "inactive") {
                mediaRecorder.start(1000);
            }
        });
}

rec.start();
