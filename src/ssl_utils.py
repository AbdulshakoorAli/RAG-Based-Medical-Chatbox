import os
import ssl


def configure_ssl_environment():
    """Ensure SSL certificate environment is valid for httpx/ollama imports."""
    ssl_cert_file = os.environ.get("SSL_CERT_FILE")

    if ssl_cert_file:
        if os.path.exists(ssl_cert_file):
            return ssl_cert_file

        os.environ.pop("SSL_CERT_FILE", None)

    try:
        ssl.create_default_context()
    except Exception:
        return None

    return None
