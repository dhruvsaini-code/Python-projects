import io
import pandas as pd
from typing import Optional

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Converts a pandas DataFrame into raw CSV bytes (UTF-8 encoded)."""
    return df.to_csv(index=False).encode('utf-8')

def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "IPL_Data") -> Optional[bytes]:
    """
    Converts a pandas DataFrame into Excel binary bytes using openpyxl.
    Returns None if openpyxl is not installed.
    """
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        return buffer.getvalue()
    except ImportError:
        # Gracefully handle environment where openpyxl is missing
        print("Excel exporter failed: openpyxl is not installed in the workspace environment.")
        return None
    except Exception as e:
        print(f"Excel exporter error: {e}")
        return None
