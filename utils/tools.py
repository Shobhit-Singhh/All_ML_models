import os 
import sys
import time
import logging

def log():
    print()

def path():
    print(os.path.dirname(os.path.abspath(__file__)))
    print(os.path.abspath(__file__))
    print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))