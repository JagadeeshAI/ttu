#!/usr/bin/env python3
import sys
import os

# Ensure the root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unlearning.utils import unlearn_main

if __name__ == "__main__":
    unlearn_main(forget_name="Sneha Singh")
