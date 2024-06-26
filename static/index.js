function uploadVideo()
{
    var formData = new FormData();
    var fileInput = document.getElementById('video-upload');
    var file = fileInput.files[0];
    formData.append('video', file);

    var loader = document.getElementById('loader');
    loader.style.display = 'flex';

    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/upload', true);
    xhr.onreadystatechange = function () {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            loader.style.display = 'none'; // Hide the loader when the request is done
            if (xhr.status === 200) {
                var response = JSON.parse(xhr.responseText);
                document.getElementById('video-source').setAttribute('src', response.video_path);
                document.getElementById('video-preview').load();
                if (response.prediction >= 0.5) {
                    document.getElementById('prediction-result-class').innerText = 'Video is Forged - ' + response.prediction;
                } else {
                    document.getElementById('prediction-result-class').innerText = 'Video is Original - ' + response.prediction;
                }
                document.getElementById('video-result').style.display = 'block';

            } else {
                console.error('Error:', xhr.status);
            }
        }
    };
    xhr.send(formData);
}