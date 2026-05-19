# ==============================================================================
# models/__init__.py — Factory function tạo TransVG model
# ==============================================================================

from .model import TransVG


def build_model(config):
    """
    Tạo TransVG model từ config.

    Cách dùng:
        from models import build_model
        model = build_model(config)
    """
    return TransVG(config)
