function uploadImage() {
    const input = document.getElementById('imageInput');
    const file = input.files[0];
    if (!file) {
        alert("Pilih gambar dulu!");
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    const originalImgUrl = URL.createObjectURL(file);

    fetch('http://localhost:5000/process-image', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.text().then(text => { throw new Error(text); });
        }
        return response.blob();
    })
    .then(blob => {
        const imgUrl = URL.createObjectURL(blob);
        document.getElementById('resultImage').innerHTML = `
            <div style="display: flex; flex-direction: row">
                <img src="${originalImgUrl}" style="
                    display: inline-block; 
                    padding: 20px; 
                    background-color: #000000;
                    border-radius: 10px;
                    margin: 50px 20px"/>
                <img src="${imgUrl}" style="
                    display: inline-block; 
                    padding: 20px; 
                    background-color: #000000;
                    border-radius: 10px;
                    margin: 50px 20px"/>
            </div>
            `;
    })
    .catch(err => {
        alert("Gagal memproses gambar.");
        console.error(err);
    });
}
