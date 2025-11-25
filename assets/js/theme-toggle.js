// Theme toggle functionality
(function() {
  const THEME_KEY = 'site-theme';
  const DARK_THEME = 'dark';
  const LIGHT_THEME = 'light';

  // Get saved theme or default to dark
  function getSavedTheme() {
    return localStorage.getItem(THEME_KEY) || DARK_THEME;
  }

  // Apply theme to document
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_KEY, theme);

    // Update toggle button icon
    const toggleBtn = document.getElementById('theme-toggle');
    if (toggleBtn) {
      toggleBtn.innerHTML = theme === DARK_THEME ? '☀️' : '🌙';
      toggleBtn.setAttribute('aria-label', theme === DARK_THEME ? 'Switch to light mode' : 'Switch to dark mode');
    }
  }

  // Toggle between themes
  function toggleTheme() {
    const currentTheme = getSavedTheme();
    const newTheme = currentTheme === DARK_THEME ? LIGHT_THEME : DARK_THEME;
    applyTheme(newTheme);
  }

  // Initialize theme on page load
  document.addEventListener('DOMContentLoaded', function() {
    applyTheme(getSavedTheme());

    // Add click handler to toggle button
    const toggleBtn = document.getElementById('theme-toggle');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', toggleTheme);
    }
  });

  // Apply theme immediately (before DOM loads) to prevent flash
  applyTheme(getSavedTheme());
})();
