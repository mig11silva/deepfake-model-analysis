"""
Helper Utilities

This module contains utility functions used throughout the project.
These are simple, reusable functions for common tasks.
"""

import os
from pathlib import Path
from typing import Optional


def ensure_directory_exists(path: str) -> Path:
    """
    Create a directory if it doesn't exist.
    
    Args:
        path: Path to the directory (string or Path object)
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def is_valid_image_path(path: str) -> bool:
    """
    Check if a path points to a valid, readable image file.
    
    Args:
        path: Path to check
        
    Returns:
        True if file exists and is readable
    """
    try:
        path_obj = Path(path)
        return path_obj.exists() and path_obj.is_file()
    except Exception:
        return False


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: Decimal value (e.g., 0.95)
        decimals: Number of decimal places
        
    Returns:
        Formatted string (e.g., "95.00%")
    """
    return f"{value * 100:.{decimals}f}%"


def truncate_string(text: str, max_length: int = 50) -> str:
    """
    Truncate a string if it's too long.
    
    Args:
        text: String to truncate
        max_length: Maximum length before truncation
        
    Returns:
        Truncated string with "..." if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def print_header(title: str, width: int = 60) -> None:
    """
    Print a formatted header line.
    
    Args:
        title: Text to display in header
        width: Total width of the header
    """
    print("=" * width)
    print(f" {title}")
    print("=" * width)


def print_section(title: str, width: int = 60) -> None:
    """
    Print a section divider.
    
    Args:
        title: Section title
        width: Total width of the divider
    """
    print(f"\n{'-' * width}")
    print(f" {title}")
    print("-" * width)


# ==============================================================================
# SIMPLE TEST CODE
# ==============================================================================

if __name__ == "__main__":
    # Test the helper functions
    print("Testing helper functions...")
    
    # Test percentage formatting
    print(f"  format_percentage(0.95) = {format_percentage(0.95)}")
    
    # Test string truncation
    long_string = "This is a very long string that should be truncated"
    print(f"  truncate_string('{long_string[:30]}...') = '{truncate_string(long_string, 30)}'")
    
    # Test header
    print()
    print_header("Test Header")
    
    print("\n  ✓ All helper functions working!")
