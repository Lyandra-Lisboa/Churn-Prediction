import logging

logging.basicConfig(level=logging.INFO)

def alert(message: str):
    """
    Sistema simples de alertas
    """
    logging.warning(f"[ALERT] {message}")
