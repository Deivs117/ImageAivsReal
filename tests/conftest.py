import sys
import os

# Add proto/generated to path so gRPC stubs can be imported in tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'proto', 'generated'))
