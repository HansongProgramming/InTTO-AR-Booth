function refreshFrame() {
  const img = document.getElementById("video-frame");
  img.src = `/frame?cachebust=${new Date().getTime()}`;
}

setInterval(refreshFrame, 100);

function takePhoto() {
  fetch("/take_photo", {
    method: "POST"
  })
    .then(res => res.json())
    .then(data => {
      alert("📸 Photo saved: " + data.path);
    });
}
