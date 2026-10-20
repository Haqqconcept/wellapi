"""
core/privacy.py
---------------
Two protections for operators who send field data to the API.

1. Request inputs are NOT stored in the usage log by default.
   Only the metadata needed for billing is kept (endpoint, credits, timing).
   To keep inputs for debugging, set  STORE_REQUEST_INPUTS=true  in .env.

2. Startup safety check on SECRET_KEY. The default development key lets anyone
   forge login tokens. In production (DEBUG=false) the server refuses to start
   with it. In development (DEBUG=true) it only warns.

   Generate a real key with:
       python -c "import secrets; print(secrets.token_hex(32))"
   then set SECRET_KEY and DEBUG=false in your Railway variables (or .env).
"""
import json
import warnings
from typing import Any, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from core.config import settings

DEFAULT_SECRET = "dev-secret-key-change-in-production"


class _PrivacySettings(BaseSettings):
    STORE_REQUEST_INPUTS: bool = False
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


privacy = _PrivacySettings()


def input_log(payload: Any) -> Optional[str]:
    """What goes into usage_logs.input_params: the inputs if enabled, else nothing."""
    return json.dumps(payload) if privacy.STORE_REQUEST_INPUTS else None


def check_production_settings(secret_key: Optional[str] = None, debug: Optional[bool] = None) -> None:
    secret = settings.SECRET_KEY if secret_key is None else secret_key
    dbg = settings.DEBUG if debug is None else debug
    if secret == DEFAULT_SECRET:
        msg = ("SECRET_KEY is still the default development value, so login tokens can be forged. "
               "Set SECRET_KEY to a long random string.")
        if dbg:
            warnings.warn(msg + " (Allowed only because DEBUG=true.)", stacklevel=2)
        else:
            raise RuntimeError(msg + " Refusing to start with DEBUG=false.")
    elif len(secret) < 32:
        warnings.warn("SECRET_KEY is shorter than 32 characters; use a longer random value.", stacklevel=2)


check_production_settings()   # runs once, when the first router imports this module
