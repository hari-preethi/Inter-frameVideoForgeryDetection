function uploadVideo() {
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
                document.getElementById('prediction-result').innerText = response.prediction;
                if (response.prediction >= 0.5) {
                    document.getElementById('prediction-result-class').innerText = 'Forged';
                } else {
                    document.getElementById('prediction-result-class').innerText = 'Original';
                }
                document.getElementById('video-result').style.display = 'block';

                setTimeout(function() {
                    document.getElementById('shap-plot').setAttribute('src', response.shap_plot);
                    document.getElementById('feature-plot').setAttribute('src', response.feature_plot);
                    document.getElementById('shap-feature-plots').style.display = 'block';
                }, 500); // Adjust the delay if necessary
            } else {
                console.error('Error:', xhr.status);
            }
        }
    };
    xhr.send(formData);
}