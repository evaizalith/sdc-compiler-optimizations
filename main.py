import torch
import sys

# Returns a list of all compiler optimizations
# optsFile should be a file that contains 1 argument per line
def getOptimizations(optsFile):

    try:
        file = open(optsFile, 'r')
    except:
        print("Unable to open", optsFile)
        sys.exit()

    print("Reading in", optsFile)
    argsList = []
    for line in file:
        argsList.append(line)

    return argsList

def main():
    if len(sys.argv) == 1:
        optsFile = 'opts.txt'
        output = 'output.txt'
    elif len(sys.argv) == 2:
        optsFile = sys.argv[1]
        output = 'output.txt'

    LLFI_ROOT = "~/"
    LLVM_ROOT = "~/llvm-project/build/bin/"

    if torch.cuda.is_available():
        DEVICE = "cuda"
        torch.cuda.empty_cache()
    else:
        DEVICE = "cpu"

    print("Device is", DEVICE)

    opts = getOptimizations(optsFile)

if __name__ == "__main__":
    main()
