import torch
import sys
import matplotlib.pyplot as plt
from IPython import display
from model import sdcAgent

def plot(scores, meanScores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Model Training')
    plt.xlabel('Iterations')
    plt.ylabel('SDC Rate')
    plt.plot(scores)
    plt.plot(meanScores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(meanScores)-1, meanScores[-1], str(meanScores[-1]))

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

    file.close()

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
        DEVICE = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        DEVICE = torch.device("cpu")

    print("Device is", DEVICE)

    opts = torch.tensor(getOptimizations(optsFile))

    model = sdcModel()

    if exists ('trained_model.pt'):
        model.load_state_dict(torch.load('trained_model.pt'))
        print("Loading existing model")
    else:
        print("No existing model found; training from scratch")

    model = model.to(DEVICE)

    for epoch in range(10):
        print("Epoch ", epoch)

        errorRate = sdcModel.train(model, dl_train, optimizer, criterion, DEVICE)
        print("Error rate = ", errorRate)

        print()

    torch.save(model.state_dict(), "trained_model.pt")

    print("All done!")

if __name__ == "__main__":
    main()
