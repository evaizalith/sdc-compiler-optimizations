import torch
import sys

# Returns a list of all compiler optimizations
def getOptimizations(optsFile):
    print("Reading in", optsFile)
    return []

def main():
    if len(sys.argv) == 1:
        optsFile = 'opts.txt'
        output = 'output.txt'

    if torch.cuda.is_available():
        DEVICE = "cuda"
        torch.cuda.empty_cache()
    else:
        DEVICE = "cpu"

    print("Device is", DEVICE)

    opts = getOptimizations(optsFile)

if __name__ == "__main__":
    main()
