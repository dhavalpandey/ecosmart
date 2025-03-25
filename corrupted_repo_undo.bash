mv -v .git .git_old &&            # Remove old Git files
git init &&                       # Initialise new repository
git remote add origin "https://github.com/dhavalpandey/ecosmart.git" && # Link to old repository
git fetch &&                      # Get old history
# Note that some repositories use 'master' in place of 'main'. Change the following line if your remote uses 'master'.
git reset origin/main --mixed     # Force update to old history.