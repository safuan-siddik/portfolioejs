// Wait for the DOM to be fully loaded
document.addEventListener("DOMContentLoaded", function () {
  // Mobile menu functionality
  const hamburger = document.querySelector(".hamburger");
  const mobileMenu = document.querySelector(".mobile-menu");

  if (hamburger && mobileMenu) {
    hamburger.addEventListener("click", () => {
      mobileMenu.classList.toggle("isActive");
    });
  }

  // Search functionality
  const searchToggle = document.querySelector('[data-toggle="search"]');
  const searchBar = document.getElementById("searchBar");
  const searchForm = document.querySelector(".forum-search form");
  const searchInput = document.querySelector(
    '.forum-search input[type="text"]'
  );

  if (searchToggle && searchBar) {
    // Toggle search bar
    searchToggle.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();

      const isHidden =
        searchBar.style.display === "none" || !searchBar.style.display;
      searchBar.style.display = isHidden ? "block" : "none";
      searchToggle.classList.toggle("active", isHidden);

      if (isHidden && searchInput) {
        setTimeout(() => {
          searchInput.focus();
        }, 100);
      }
    });

    // Handle search form submission
    if (searchForm) {
      searchForm.addEventListener("submit", (e) => {
        const query = searchInput.value.trim();
        if (!query) {
          e.preventDefault();
          return;
        }
      });
    }

    // Close search bar when clicking outside
    document.addEventListener("click", (e) => {
      if (!searchBar.contains(e.target) && !searchToggle.contains(e.target)) {
        searchBar.style.display = "none";
        searchToggle.classList.remove("active");
      }
    });

    // Prevent search bar from closing when clicking inside it
    searchBar.addEventListener("click", (e) => {
      e.stopPropagation();
    });
  }

  // Add smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute("href"));
      if (target) {
        target.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    });
  });

  // Add animation on scroll
  const animateOnScroll = () => {
    const elements = document.querySelectorAll(".animate-on-scroll");
    elements.forEach((element) => {
      const elementTop = element.getBoundingClientRect().top;
      const elementBottom = element.getBoundingClientRect().bottom;
      const isVisible = elementTop < window.innerHeight && elementBottom >= 0;

      if (isVisible) {
        element.classList.add("visible");
      }
    });
  };

  // Initial check for elements in view
  animateOnScroll();

  // Check for elements in view on scroll
  window.addEventListener("scroll", animateOnScroll);

  // Handle error and success messages
  const messageContainer = document.getElementById("messageContainer");
  const errorMessage = messageContainer?.querySelector(".error-message");
  const successMessage = messageContainer?.querySelector(".success-message");

  // Function to show a message
  const showMessage = (message) => {
    if (message) {
      message.classList.add("visible");
      setTimeout(() => {
        message.classList.remove("visible");
        // Wait for fade out animation to complete before removing
        setTimeout(() => {
          message.remove();
          // If no messages left, remove the container
          if (messageContainer && !messageContainer.hasChildNodes()) {
            messageContainer.remove();
          }
        }, 300); // Match the CSS transition duration
      }, 3000);
    }
  };

  // Only show messages if they exist and have content
  if (errorMessage && errorMessage.textContent.trim()) {
    showMessage(errorMessage);
  }
  if (successMessage && successMessage.textContent.trim()) {
    showMessage(successMessage);
  }

  // Handle form submission messages
  const newPostForm = document.querySelector(".new-post-form form");
  if (newPostForm) {
    newPostForm.addEventListener("submit", () => {
      // Remove any existing messages
      if (messageContainer) {
        messageContainer.remove();
      }
    });
  }
});
