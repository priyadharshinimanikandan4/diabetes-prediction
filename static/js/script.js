// Add interactivity here
document.addEventListener("DOMContentLoaded", function () {
    // Example: Add a confirmation dialog before deleting a record
    const deleteButtons = document.querySelectorAll(".delete-btn");
    deleteButtons.forEach((button) => {
        button.addEventListener("click", (e) => {
            if (!confirm("Are you sure you want to delete this record?")) {
                e.preventDefault();
            }
        });
    });
});