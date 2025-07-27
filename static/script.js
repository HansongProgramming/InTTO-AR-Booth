  function refreshFrame() {
    const img = document.getElementById("video-frame");
    img.src = `/frame?cachebust=${new Date().getTime()}`;
  }
  setInterval(refreshFrame, 100);

  function handlePhotoClick() {
    const delay = parseInt(document.getElementById("timer-select").value);
    if (delay === 0) {
      takePhoto();
      return;
    }

    const countdownEl = document.getElementById("countdown");
    let secondsLeft = delay / 1000;
    countdownEl.textContent = secondsLeft;
    countdownEl.style.display = "block";

    const countdownInterval = setInterval(() => {
      secondsLeft--;
      if (secondsLeft <= 0) {
        clearInterval(countdownInterval);
        countdownEl.style.display = "none";
        takePhoto();
      } else {
        countdownEl.textContent = secondsLeft;
      }
    }, 1000);
  }

  function takePhoto() {
    fetch("/take_photo", { method: "POST" })
      .then(res => res.json())
      .then(data => {
        alert("📸 Photo saved: " + data.path);
      });
  }