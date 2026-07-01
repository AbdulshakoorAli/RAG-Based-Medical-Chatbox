import os
import tempfile
import unittest

from src.ssl_utils import configure_ssl_environment


class ConfigureSslEnvironmentTests(unittest.TestCase):
    def test_invalid_ssl_cert_file_does_not_remain_in_environment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_path = os.path.join(tmp_dir, "missing-cert.pem")
            previous_value = os.environ.get("SSL_CERT_FILE")
            os.environ["SSL_CERT_FILE"] = invalid_path

            try:
                configured_path = configure_ssl_environment()
                self.assertTrue(configured_path is None or os.path.exists(configured_path))
                self.assertNotEqual(os.environ.get("SSL_CERT_FILE"), invalid_path)
            finally:
                if previous_value is None:
                    os.environ.pop("SSL_CERT_FILE", None)
                else:
                    os.environ["SSL_CERT_FILE"] = previous_value


if __name__ == "__main__":
    unittest.main()
