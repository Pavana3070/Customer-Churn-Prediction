import os

# Render dynamically assigns a PORT environment variable
port = os.environ.get('PORT', '10000')
bind = f"0.0.0.0:{port}"

# Increase timeout for Machine Learning training tasks
timeout = 120

# Limit workers to 1 to prevent Out-Of-Memory (OOM) errors on Render's 512MB Free Tier
workers = 1
