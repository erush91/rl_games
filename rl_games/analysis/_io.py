# MIT License

# Copyright (c) 2020 Federico Claudi

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# https://github.com/FedeClaudi/pyrnn/blob/a3b6a32b0d00d8d732c484f4a35cf581f345b4b0/pyrnn/_io.py

import json
from pathlib import Path

def save_json(filepath,
              content,
              append=False,
              topcomment=None
):
    """Saves content to a json file.

    Args:
        filepath: path to a file (must include .json)
        content: dictionary of stuff to save
    """
    fp = Path(filepath)
    if fp.suffix not in ('.json'):
        raise ValueError(f"Filepath {fp} not valid should point to json file")

    with open(filepath, 'w') as json_file:
        json.dump(content, json_file, indent=4, default=encode_complex)

def load_json(filepath):
    """Loads a json file.
    
    Args:
        filepath: path to json file
    """
    fp = Path(filepath)
    if not fp.exists():
        raise ValueError("Unrecognized file path: {}".format(filepath))

    with open(filepath) as f:
        data = json.load(f, object_hook=decode_complex)

    return data

def encode_complex(dict):
    """Encodes complex numbers to JSON.
    
    https://www.machinelearningplus.com/python-json-guide/
    https://realpython.com/python-json/
    
    Args:
        object: input data
    """
    if isinstance(dict, complex):
        return ['__complex__', dict.real, dict.imag]
    raise TypeError(repr(dict) + " is not JSON serialized")

def decode_complex(dict):
    """Decodes complex numbers from JSON.
    
    https://www.machinelearningplus.com/python-json-guide/
    https://realpython.com/python-json/
    
    Args:
        object: input data
    """
    if '__complex__' in dict:
        return complex(dict['real'], dict['imag'])
    return dict