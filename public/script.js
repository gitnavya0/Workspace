function handleImageUpload(file) {
    const formData = new FormData();
    formData.append('image', file);

    fetch('http://localhost:1000/upload', { // Backend upload endpoint
        method: 'POST',
        body: formData3
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log('Uploaded successfully:', data);
                console.log('File URL:', data.webViewLink);
                // Use the data.webViewLink for further processing
            } else {
                alert('Error uploading file');
            }
        })
        .catch(error => console.error('Error:', error));
}
