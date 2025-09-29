import psychohistory

def test_import():
    """Verify the psychohistory package imports correctly"""
    assert hasattr(psychohistory, "__version__"), "Version attribute missing"
    assert psychohistory.__version__ == "0.1.0"
