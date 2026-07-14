# Package entrypoint does not import app or database modules at import time.
# This prevents early evaluation of environment-specific settings during test discovery.
