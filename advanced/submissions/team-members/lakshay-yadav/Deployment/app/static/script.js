const uploadBox = document.getElementById("uploadBox");
const fileInput = document.getElementById("fileInput");
const previewImage = document.getElementById("previewImage");
const analyzeBtn = document.getElementById("analyzeBtn");
const statusText = document.getElementById("statusText");
const resultPanel = document.getElementById("resultPanel");
const confidenceText = document.getElementById("confidenceText");
const resultLabel = document.getElementById("resultLabel");
const loaderContainer = document.getElementById("loaderContainer");

let brainAnim;

// 🧠 Load rotating brain animation
function loadBrainAnimation() {
  loaderContainer.innerHTML = "";
  brainAnim = lottie.loadAnimation({
    container: loaderContainer,
    renderer: "svg",
    loop: true,
    autoplay: true,
    path: "https://assets4.lottiefiles.com/packages/lf20_vf3krn3j.json"
  });
}

uploadBox.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", handleFile);

uploadBox.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadBox.style.backgroundColor = "rgba(255,255,255,0.1)";
});
uploadBox.addEventListener("dragleave", () => {
  uploadBox.style.backgroundColor = "";
});
uploadBox.addEventListener("drop", (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  fileInput.files = e.dataTransfer.files;
  handleFile({ target: { files: [file] } });
});

function handleFile(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    previewImage.style.display = "block";
  };
  reader.readAsDataURL(file);
  analyzeBtn.disabled = false;
  statusText.textContent = "Ready to analyze.";
}

analyzeBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append("file", file);

  resultPanel.classList.remove("visible");
  resultPanel.classList.add("hidden");
  loadBrainAnimation();

  statusText.textContent = "Analyzing image...";
  analyzeBtn.disabled = true;

  try {
    const response = await fetch("/predict", { method: "POST", body: formData });
    const result = await response.json();

    if (response.status !== 200) {
      updateError(result.error || "Invalid image. Please upload a valid MRI scan.");
    } else {
      updateResult(result);
    }
  } catch {
    statusText.textContent = "Error during analysis.";
  }

  analyzeBtn.disabled = false;
});

function animateConfidence(finalValue) {
  let start = 0;
  const duration = 1500;
  const increment = finalValue / (duration / 16);

  const timer = setInterval(() => {
    start += increment;
    if (start >= finalValue) {
      start = finalValue;
      clearInterval(timer);
    }
    confidenceText.textContent = `Confidence: ${start.toFixed(2)}%`;
  }, 16);
}

function updateResult(data) {
  const confidence = parseFloat(data.confidence);
  const label = data.prediction;

  if (brainAnim) brainAnim.destroy();

  resultLabel.textContent =
    label === "Brain Tumor Detected"
      ? `🧠 Brain Tumor Detected (High Confidence)`
      : `✅ No Tumor Detected (Low Confidence)`;

  confidenceText.textContent = "Confidence: 0%";
  animateConfidence(confidence);

  resultPanel.classList.remove("hidden");
  setTimeout(() => resultPanel.classList.add("visible"), 50);
  statusText.textContent = "Analysis complete.";
}

// 🚫 Handle invalid image uploads gracefully
function updateError(message) {
  if (brainAnim) brainAnim.destroy();
  resultLabel.textContent = `⚠️ ${message}`;
  confidenceText.textContent = "";
  resultPanel.classList.remove("hidden");
  resultPanel.classList.add("visible");
  statusText.textContent = "Upload a valid MRI scan.";
}
