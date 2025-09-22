// Timer
function startTimer() {
    if (!timerId) {
        timerId = setInterval(() => {
            count++;
            imageUploadInfo.innerText = `Memproses gambar: ${count} detik`;
        }, 1000);
    }
}

function stopTimer() {
    clearInterval(timerId);
    timerId = null;
    console.log(`Proses terjemahan memakan waktu: ${count} detik`);
    count = 0;
}

function resetTimer() {
    imageUploadInfo.innerText = `Proses terjemahan selesai! \nMemakan waktu: ${count} detik`;
    imageUploadInfo.style.backgroundColor = "rgba(0, 128, 0, 0.6)";
    stopTimer();
}