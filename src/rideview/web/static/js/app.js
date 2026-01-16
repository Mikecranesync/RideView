/**
 * RideView - Frontend JavaScript
 *
 * Handles:
 * - Status polling and display updates
 * - Video feed controls
 * - Snapshot capture
 */

// Configuration
const POLL_INTERVAL = 500; // ms

// State
let statusPollTimer = null;

/**
 * Initialize the application
 */
function init() {
    console.log('RideView initialized');
    startStatusPolling();
}

/**
 * Start polling for status updates
 */
function startStatusPolling() {
    if (statusPollTimer) {
        clearInterval(statusPollTimer);
    }
    statusPollTimer = setInterval(fetchStatus, POLL_INTERVAL);
}

/**
 * Fetch current detection status from API
 */
async function fetchStatus() {
    try {
        const response = await fetch('/api/status');
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        updateStatusDisplay(data);
    } catch (error) {
        console.error('Error fetching status:', error);
    }
}

/**
 * Update the status display with new data
 */
function updateStatusDisplay(data) {
    if (data.status === 'no_data') {
        return;
    }

    // Update status indicator
    const indicator = document.getElementById('status-indicator');
    if (indicator) {
        indicator.textContent = data.result;
        indicator.className = 'status-indicator status-' + data.result.toLowerCase();
    }

    // Update metric values
    updateMetric('metric-status', data.result);
    updateMetric('metric-coverage', data.coverage_percent.toFixed(1) + '%');
    updateMetric('metric-gaps', data.gap_count);
    updateMetric('metric-confidence', (data.confidence * 100).toFixed(0) + '%');
    updateMetric('metric-time', data.processing_time_ms.toFixed(1) + 'ms');

    // Update reason
    const reasonEl = document.getElementById('analysis-reason');
    if (reasonEl) {
        reasonEl.textContent = data.reason || 'No details';
    }

    // Color code the status metric
    const statusMetric = document.getElementById('metric-status');
    if (statusMetric) {
        statusMetric.className = 'metric-value status-' + data.result.toLowerCase();
    }
}

/**
 * Update a single metric element
 */
function updateMetric(elementId, value) {
    const el = document.getElementById(elementId);
    if (el) {
        el.textContent = value;
    }
}

/**
 * Format a number with fixed decimal places
 */
function formatNumber(value, decimals = 1) {
    return Number(value).toFixed(decimals);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
