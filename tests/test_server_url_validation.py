"""Tests for outbound MCP/Ollama destination validation."""

import socket
import unittest
from unittest.mock import patch

from src.mcp_client import ensure_valid_server_url


def resolved_to(address: str):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 80))]


class ServerUrlValidationTests(unittest.TestCase):
    @patch("src.mcp_client.socket.getaddrinfo", return_value=resolved_to("127.0.0.1"))
    def test_loopback_remains_available_for_local_mcp(self, _resolver):
        ensure_valid_server_url("http://localhost:8000/sse")

    @patch("src.mcp_client.socket.getaddrinfo", return_value=resolved_to("10.0.0.5"))
    def test_private_network_requires_operator_opt_in(self, _resolver):
        with self.assertRaises(ValueError):
            ensure_valid_server_url("http://internal.example/sse", allow_private=False)
        ensure_valid_server_url("http://internal.example/sse", allow_private=True)

    @patch("src.mcp_client.socket.getaddrinfo", return_value=resolved_to("169.254.169.254"))
    def test_cloud_metadata_is_always_blocked(self, _resolver):
        with self.assertRaises(ValueError):
            ensure_valid_server_url("http://metadata.example/latest", allow_private=True)

    @patch("src.mcp_client.socket.getaddrinfo", return_value=resolved_to("93.184.216.34"))
    def test_public_https_url_is_allowed(self, _resolver):
        ensure_valid_server_url("https://mcp.example/sse")

    def test_embedded_credentials_are_rejected(self):
        with self.assertRaises(ValueError):
            ensure_valid_server_url("https://user:password@mcp.example/sse")


if __name__ == "__main__":
    unittest.main()
