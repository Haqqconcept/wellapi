"""
core/config.py
--------------
Application configuration loaded from environment variables.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "WellAPI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    DATABASE_URL: str = "sqlite:///./wellapi.db"
    SECRET_KEY: str = "dev-secret-key-change-in-production"

    STRIPE_SECRET_KEY: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""
    PAYSTACK_SECRET_KEY: str = ""

    FREE_CREDITS_ON_SIGNUP: int = 50

    # Pricing (credits per call)
    ML_CALL_COST: int = 1       # 1 credit per ML prediction
    PHYSICS_CALL_COST: int = 4  # 4 credits per physics traverse

    # Credit bundles: {bundle_id: (credits, price_usd)}
    BUNDLES: dict = {
        "starter"     : (100,  5.00),
        "basic"       : (500,  20.00),
        "professional": (2000, 75.00),
        "enterprise"  : (8000, 250.00),
        "academic"    : (1200, 30.00),
    }

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
