let selectedFiles = [];
let allImages = [];
let imageUploadInfo;

let count = 0;
let timerId = null;

let input, inputPDF, inputTranslate;

// Event binding setelah DOM siap
document.addEventListener('DOMContentLoaded', () => {
    input = document.getElementById('imageInput');
    inputPDF = document.getElementById('downloadPdfBtn');
    inputTranslate = document.getElementById('button-translate');
    imageUploadInfo = document.getElementById('imageInfo');
    theResult = document.getElementsByClassName('results');

    input.addEventListener('change', (event) => {
        if (event.target.files.length > 0) {
            handleUpload(event.target.files);
        }
        // kalau cancel (files.length = 0), tidak lakukan apa-apa
    });

    inputTranslate.addEventListener('click', () => {
        translateImage();
    });

    inputPDF.addEventListener('click', () => {
        downloadPDF(allImages);
    });
});

// Fungsi upload gambar
function handleUpload(inputFiles) {
    if (!inputFiles || inputFiles.length === 0) {
        return; // jangan tampilkan alert saat cancel
    }

    selectedFiles = Array.from(inputFiles);

    imageUploadInfo.innerText = `Jumlah gambar yang diunggah: ${selectedFiles.length}`;
    imageUploadInfo.style.backgroundColor = "rgba(0, 0, 0, 0.6)";
    imageUploadInfo.style.zIndex = 10;

    // reset gambar
    document.getElementById('imageInput').value = "";
}

// Fungsi buat penerjemah
function translateImage() {
    if (selectedFiles.length === 0) {
        alert("Upload gambar dulu sebelum menerjemahkan!");
        return;
    }

    let sourceValue = document.getElementById("source-lang").value;
    let targetValue = document.getElementById("target-lang").value;

    if (!sourceValue || !targetValue) {
        alert("Pilih bahasa sumber dan bahasa target!");
        return;
    }

    startTimer();
    setButtonDisabled(true);
    imageUploadInfo.style.backgroundColor = "rgba(0, 0, 0, 0.6)";
    imageUploadInfo.innerText = "Sedang memproses...";

    const formData = new FormData();
    for (let i = 0; i < selectedFiles.length; i++) {
        formData.append('image[]', selectedFiles[i]);
    }
    formData.append('source', sourceValue);
    formData.append('target', targetValue);

    const preview = selectedFiles.map(file => URL.createObjectURL(file));

    // fetch('https://ceb885f109c9.ngrok-free.app/process-image', {
    fetch('http://127.0.0.1:5000/process-image', {
        method: 'POST',
        body: formData,
    })
    .then(response => {
        if (!response.ok) {
            stopTimer();
            imageUploadInfo.innerText = `Terdapat kesalahan!`;
            imageUploadInfo.style.backgroundColor = "rgba(255, 0, 0, 0.6)";
            return response.text().then(text => { throw new Error(text); });
        }
        return response.json();
    })
    .then(data => {
        if (!data.results || data.results.length === 0) {
            throw new Error("Hasil terjemahan kosong.");
        }

        html = `<div style="display: flex; flex-wrap: wrap; justify-content: center; margin: 50px auto">`;
        allImages = [];

        for (let i = 0; i < preview.length; i++) {
            let resultUrl = data.results[i] || ""; // fallback
            html += `
                <div style="margin: 10px auto; display: flex; flex-wrap: wrap; justify-content: space-evenly">
                    <div style = "max-width: 720px; padding: 20px; background-color: #000000; border-radius: 10px; margin: 10px; display: flex; flex-direction: column; gap: 10px">
                        <p style="color: #ffffff; text-align: center">Asli</p>
                        <img src="${preview[i]}"/>
                    </div>

                    <div style = "max-width: 720px; padding: 20px; background-color: #000000; border-radius: 10px; margin: 10px; display: flex; flex-direction: column; gap: 10px">
                        <p style="color: #ffffff; text-align: center">Terjemahan</p>
                        ${resultUrl ? `
                        <img src="${resultUrl}"/>` : ""}
                    </div>
                </div>
            `;
            if (resultUrl) {
                allImages.push(resultUrl);
            }
        }
        html += `</div>`;
        document.getElementById('resultImage').innerHTML = html;
        resetTimer();
        setButtonDisabled(false);
    })
    .catch(err => {
        stopTimer();
        imageUploadInfo.innerText = "Terjadi kesalahan!";
        imageUploadInfo.style.backgroundColor = "rgba(255, 0, 0, 0.6)";
        alert("Gagal memproses gambar.");
        console.error(err);
        setButtonDisabled(false);
    });
}

// Fungsi download gambar ke PDF
function downloadPDF(imageUrls) {
    if (!imageUrls || imageUrls.length === 0) {
        alert("Belum ada gambar untuk di-download!");
        return;
    }

    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF();

    let promises = imageUrls.map(url => {
        return new Promise((resolve) => {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.src = url;
            img.onload = () => resolve(img);
            img.onerror = () => resolve(null);
        });
    });

    Promise.all(promises).then(images => {
        images.forEach((img, index) => {
            if (!img) return; // skip kalau gagal load
            const pageWidth = pdf.internal.pageSize.getWidth();
            const pageHeight = pdf.internal.pageSize.getHeight();

            // scale agar gambar muat di halaman
            let ratio = Math.min(pageWidth / img.width, pageHeight / img.height);
            let w = img.width * ratio;
            let h = img.height * ratio;

            let x = (pageWidth - w) / 2;
            let y = (pageHeight - h) / 2;

            if (index > 0) pdf.addPage();
            pdf.addImage(img, 'JPEG', x, y, w, h);
        });
        pdf.save("hasil_gambar.pdf");
    });
}
