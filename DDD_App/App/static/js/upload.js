(function () {
    var input = document.getElementById("file");
    var dropzone = document.getElementById("dropzone");
    var preview = document.getElementById("preview");
    var filename = document.getElementById("filename");
    var copy = document.getElementById("dropzone-copy");
    var submit = document.getElementById("submit");

    if (!input || !dropzone) {
        return;
    }

    function showFile(file) {
        if (!file) {
            preview.hidden = true;
            preview.removeAttribute("src");
            filename.hidden = true;
            filename.textContent = "";
            copy.hidden = false;
            submit.disabled = true;
            dropzone.classList.remove("has-file");
            return;
        }

        filename.textContent = file.name;
        filename.hidden = false;
        copy.hidden = true;
        submit.disabled = false;
        dropzone.classList.add("has-file");

        var url = URL.createObjectURL(file);
        preview.src = url;
        preview.alt = "Selected photo preview";
        preview.hidden = false;
    }

    input.addEventListener("change", function () {
        showFile(input.files && input.files[0]);
    });

    ["dragenter", "dragover"].forEach(function (eventName) {
        dropzone.addEventListener(eventName, function (event) {
            event.preventDefault();
            dropzone.classList.add("is-dragover");
        });
    });

    ["dragleave", "drop"].forEach(function (eventName) {
        dropzone.addEventListener(eventName, function (event) {
            event.preventDefault();
            dropzone.classList.remove("is-dragover");
        });
    });

    dropzone.addEventListener("drop", function (event) {
        var files = event.dataTransfer && event.dataTransfer.files;
        if (!files || !files.length) {
            return;
        }

        var transfer = new DataTransfer();
        transfer.items.add(files[0]);
        input.files = transfer.files;
        showFile(files[0]);
    });
})();
