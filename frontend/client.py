"""
client.py - thin Python client for the WellAPI server.

Kept separate from the Streamlit UI so it can be tested on its own and reused
in scripts or notebooks:

    from client import WellAPIClient
    c = WellAPIClient("http://127.0.0.1:8000")
    c.login("me@example.com", "password")
    print(c.predict_ml({...}))
"""
from typing import Any, Dict, List, Optional
import requests


class ApiError(Exception):
    """Raised for any failed call, with a message that is safe to show a user."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def _explain(resp: requests.Response) -> str:
    """Turn a FastAPI error body into one readable sentence."""
    try:
        detail = resp.json().get("detail")
    except Exception:
        return f"Server returned {resp.status_code}."
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):  # pydantic validation errors
        parts = []
        for item in detail:
            loc = ".".join(str(p) for p in item.get("loc", []) if p not in ("body",))
            parts.append(f"{loc}: {item.get('msg', 'invalid')}" if loc else item.get("msg", "invalid"))
        return "; ".join(parts) or "Invalid input."
    return f"Server returned {resp.status_code}."


class WellAPIClient:
    def __init__(self, base_url: str, token: Optional[str] = None,
                 api_key: Optional[str] = None, timeout: float = 90.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.api_key = api_key
        self.timeout = timeout

    # ── plumbing ──────────────────────────────────────────────────────────────
    def _call(self, method: str, path: str, *, auth: str = "none", **kw) -> Any:
        headers = {}
        if auth == "jwt":
            if not self.token:
                raise ApiError("Please log in first.", 401)
            headers["Authorization"] = f"Bearer {self.token}"
        elif auth == "key":
            if not self.api_key:
                raise ApiError("No API key available. Create one on the Account page.", 401)
            headers["X-API-Key"] = self.api_key
        try:
            resp = requests.request(method, self.base_url + path, headers=headers,
                                    timeout=self.timeout, **kw)
        except requests.ConnectionError:
            raise ApiError(f"Cannot reach the WellAPI server at {self.base_url}. Is it running?")
        except requests.Timeout:
            raise ApiError("The server took too long to respond.")
        if resp.status_code >= 400:
            msg = _explain(resp)
            if resp.status_code == 402 and "nsufficient" not in msg:
                msg = "Not enough credits. " + msg
            raise ApiError(msg, resp.status_code)
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    # ── server ────────────────────────────────────────────────────────────────
    def health(self) -> Dict[str, Any]:
        return self._call("GET", "/health")

    # ── auth ──────────────────────────────────────────────────────────────────
    def register(self, email: str, name: str, password: str, organisation: str = "") -> Dict[str, Any]:
        body = {"email": email, "name": name, "password": password}
        if organisation.strip():
            body["organisation"] = organisation.strip()
        return self._call("POST", "/auth/register", json=body)

    def login(self, email: str, password: str) -> str:
        data = self._call("POST", "/auth/login", data={"username": email, "password": password})
        self.token = data["access_token"]
        return self.token

    def list_keys(self) -> List[Dict[str, Any]]:
        return self._call("GET", "/auth/keys", auth="jwt")

    def create_key(self, label: str = "Frontend key") -> Dict[str, Any]:
        return self._call("POST", "/auth/keys", auth="jwt", json={"label": label})

    def delete_key(self, key_id: str) -> None:
        self._call("DELETE", f"/auth/keys/{key_id}", auth="jwt")

    def ensure_api_key(self) -> str:
        """Use the first active key, or create one. Stores it on the client."""
        active = [k for k in self.list_keys() if k.get("is_active")]
        key = active[0] if active else self.create_key()
        self.api_key = key["key"]
        return self.api_key

    # ── account ───────────────────────────────────────────────────────────────
    def me(self) -> Dict[str, Any]:
        return self._call("GET", "/account/me", auth="jwt")

    def usage(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._call("GET", "/account/usage", auth="jwt", params={"limit": limit})

    def bundles(self) -> List[Dict[str, Any]]:
        return self._call("GET", "/account/bundles")

    def topup(self, bundle: str, gateway: str) -> Dict[str, Any]:
        return self._call("POST", "/account/topup", auth="jwt",
                          json={"bundle": bundle, "gateway": gateway})

    # ── predictions (X-API-Key) ───────────────────────────────────────────────
    def predict_ml(self, well: Dict[str, float]) -> Dict[str, Any]:
        return self._call("POST", "/predict/ml", auth="key", json=well)

    def predict_physics(self, well: Dict[str, float], include_traverse: bool = True) -> Dict[str, Any]:
        return self._call("POST", "/predict/physics", auth="key", json=well,
                          params={"include_traverse": str(include_traverse).lower()})

    def nodal(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return self._call("POST", "/nodal/analyze", auth="key", json=request)
