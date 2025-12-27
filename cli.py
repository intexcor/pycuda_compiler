#!/usr/bin/env python3
import argparse
import sys
import os
from compiler import CUDAProgram

def main():
    parser = argparse.ArgumentParser(description="PyCUDA Compiler CLI - Compile Python to CUDA C++")
    parser.add_argument("input_file", help="Path to Python source file")
    parser.add_argument("-o", "--output", help="Path to output .cu file (default: stdout)", default=None)
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.", file=sys.stderr)
        sys.exit(1)
        
    try:
        with open(args.input_file, 'r') as f:
            source = f.read()
            
        print(f"Compiling '{args.input_file}'...", file=sys.stderr)
        program = CUDAProgram(source, debug=args.debug)
        
        cuda_code = program.cuda_code
        if not cuda_code:
            print("Error: No CUDA code generated. usage of @kernel or @cuda_compile in input?", file=sys.stderr)
            sys.exit(1)
            
        if args.output:
            with open(args.output, 'w') as f:
                f.write(cuda_code)
            print(f"Success! CUDA code written to '{args.output}'", file=sys.stderr)
        else:
            print(cuda_code)
            
    except Exception as e:
        print(f"Compilation Error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
