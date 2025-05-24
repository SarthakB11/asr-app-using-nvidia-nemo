document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const transcribeBtn = document.getElementById('transcribe-btn');
    const clearBtn = document.getElementById('clear-btn');
    const resultsContainer = document.getElementById('results-container');
    const modelStatus = document.getElementById('model-status');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    // State
    let currentFile = null;
    let currentFileUrl = null;
    let transcriptionResults = []; // Each result: {id, filename, transcription, timestamp, audioUrl}
    
    // Initialize theme
    initializeTheme();
    
    // Check model status
    checkModelStatus();
    
    // Event Listeners
    themeToggleBtn.addEventListener('click', toggleTheme);
    uploadArea.addEventListener('click', triggerFileInput);
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    transcribeBtn.addEventListener('click', handleTranscribe);
    clearBtn.addEventListener('click', handleClear);
    
    // Functions
    function initializeTheme() {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        document.body.className = `${savedTheme}-theme`;
    }
    
    function toggleTheme() {
        const isDarkTheme = document.body.classList.contains('dark-theme');
        const newTheme = isDarkTheme ? 'light' : 'dark';
        
        document.body.className = `${newTheme}-theme`;
        localStorage.setItem('theme', newTheme);
    }
    
    async function checkModelStatus() {
        try {
            modelStatus.classList.add('status-loading');
            modelStatus.querySelector('.status-text').textContent = 'Checking model status...';
            
            const response = await fetch('/health');
            const data = await response.json();
            
            if (data.model_loaded) {
                modelStatus.classList.remove('status-loading');
                modelStatus.classList.add('status-success');
                modelStatus.querySelector('.status-text').textContent = 'ASR Model loaded and ready';
            } else {
                modelStatus.classList.remove('status-loading');
                modelStatus.classList.add('status-error');
                modelStatus.querySelector('.status-text').textContent = 'ASR Model not loaded';
            }
        } catch (error) {
            console.error('Error checking model status:', error);
            modelStatus.classList.remove('status-loading');
            modelStatus.classList.add('status-error');
            modelStatus.querySelector('.status-text').textContent = 'Error checking model status';
        }
    }
    
    function triggerFileInput() {
        fileInput.click();
    }
    
    function handleDragOver(e) {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    }
    
    function handleDragLeave() {
        uploadArea.classList.remove('drag-over');
    }
    
    function handleDrop(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        if (e.dataTransfer.files.length > 0) {
            handleFiles(e.dataTransfer.files);
        }
    }
    
    function handleFileSelect() {
        if (fileInput.files.length > 0) {
            handleFiles(fileInput.files);
        }
    }
    
    function handleFiles(files) {
        const file = files[0]; // Take only the first file

        if (!file.type.startsWith('audio/')) {
            alert('Please select an audio file.');
            return;
        }

        currentFile = file;
        // Revoke previous file URL if any
        if (currentFileUrl) {
            URL.revokeObjectURL(currentFileUrl);
        }
        currentFileUrl = URL.createObjectURL(file);

        // Update UI to show selected file and audio player
        uploadArea.innerHTML = `
            <div class="upload-icon">
                <i class="fas fa-file-audio"></i>
            </div>
            <h3>${file.name}</h3>
            <p>${formatFileSize(file.size)}</p>
            <audio controls src="${currentFileUrl}" style="margin-top: 1rem; width: 100%; max-width: 350px;"></audio>
            <p class="file-types">Click to select a different file</p>
        `;

        // Enable buttons
        transcribeBtn.disabled = false;
        clearBtn.disabled = false;
    }
    
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    async function handleTranscribe() {
        if (!currentFile) return;

        try {
            // Show loading overlay
            loadingOverlay.classList.remove('hidden');

            const formData = new FormData();
            formData.append('file', currentFile);

            const response = await fetch('/transcribe', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Error transcribing audio');
            }

            const data = await response.json();

            // Add result to the list, including audio URL
            addTranscriptionResult(data, currentFileUrl);

            // Reset file selection
            resetFileSelection();

        } catch (error) {
            console.error('Error transcribing audio:', error);
            alert(`Error: ${error.message}`);
        } finally {
            // Hide loading overlay
            loadingOverlay.classList.add('hidden');
        }
    }
    
    function addTranscriptionResult(data, audioUrl) {
        // Create result object
        const result = {
            id: Date.now(),
            filename: data.filename,
            transcription: data.transcription,
            timestamp: new Date().toLocaleString(),
            audioUrl: audioUrl // Object URL for playback
        };

        // Add to results array (at the beginning)
        transcriptionResults.unshift(result);

        // Update UI
        updateResultsUI();
    }
    
    function updateResultsUI() {
        // Clear "No results" message if needed
        if (transcriptionResults.length > 0) {
            const noResults = resultsContainer.querySelector('.no-results');
            if (noResults) {
                noResults.remove();
            }
        }

        // Create HTML for results
        const resultsHTML = transcriptionResults.map(result => `
            <div class="result-card" data-id="${result.id}">
                <div class="result-header">
                    <span class="result-filename">${result.filename}</span>
                    <span class="result-timestamp">${result.timestamp}</span>
                </div>
                <audio controls src="${result.audioUrl}" style="width: 100%; max-width: 350px; margin-bottom: 0.5rem;"></audio>
                <div class="result-content">
                    ${result.transcription}
                </div>
            </div>
        `).join('');

        // Update container
        resultsContainer.innerHTML = resultsHTML || '<div class="no-results">No transcriptions yet</div>';
    }
    
    function handleClear() {
        resetFileSelection();
    }
    
    function resetFileSelection() {
        currentFile = null;
        if (currentFileUrl) {
            URL.revokeObjectURL(currentFileUrl);
            currentFileUrl = null;
        }
        fileInput.value = '';

        // Reset upload area
        uploadArea.innerHTML = `
            <div class="upload-icon">
                <i class="fas fa-cloud-upload-alt"></i>
            </div>
            <h3>Upload Audio File</h3>
            <p>Drag and drop an audio file here, or click to select</p>
            <p class="file-types">Supported formats: WAV, MP3, FLAC, OGG</p>
        `;

        // Disable buttons
        transcribeBtn.disabled = true;
        clearBtn.disabled = true;
    }
});
