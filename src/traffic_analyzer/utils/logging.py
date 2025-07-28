def dbg(debug: bool, msg: str = "", *args, **kwargs):
    """Печатает отладочное сообщение, если debug=True."""
    if debug:
        print(msg.format(*args, **kwargs))
