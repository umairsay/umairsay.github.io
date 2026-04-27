/* ============================================
   includes.js — injects header and footer
   into every page automatically.

   HOW IT WORKS:
   Each page has <div id="site-header"></div>
   and <div id="site-footer"></div>.
   This script fetches header.html and footer.html
   from the root and injects them.

   YOU NEVER NEED TO EDIT THIS FILE.
   Just edit header.html or footer.html.
   ============================================ */

(function () {
  // Detect if we're one level deep (i.e. in /projects/)
  const depth = window.location.pathname.split('/').filter(Boolean).length;
  const root = depth >= 2 ? '../' : '/';

  function inject(id, file) {
    const el = document.getElementById(id);
    if (!el) return;
    fetch(root + file)
      .then(r => { if (!r.ok) return ''; return r.text(); })
      .then(html => {
        if (!html) return;
        el.innerHTML = html;
        // Fix relative asset links based on depth
        if (root !== '/') {
          el.querySelectorAll('a[href^="/"]').forEach(a => {
            a.href = root + a.getAttribute('href').slice(1);
          });
        }
      })
      .catch(() => {
        // Silently fail — page still works, just no header/footer
      });
  }

  document.addEventListener('DOMContentLoaded', function () {
    inject('site-header', 'header.html');
    inject('site-footer', 'footer.html');
  });
})();
