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

    opts = getOptimizations(optsFile)

    #if exists ('trained_model.pt'):
    #    model.load_state_dict(torch.load('trained_model.pt'))
    #    print("Loading existing model")
    #else:
    #    print("No existing model found; training from scratch")

    plotScores = []
    plotMeanScores = []
    totalScore = 0
    record = 0
    agent = sdcAgent()

    # This part is essentially pseudocode
    for epoch in range(100):
        stateOld = interpretFIResults()

        action = agent.getAction(stateOld)

        selectOpts = tensorToOpts(action)
        compileAndRun(fileName, LLFI_ROOT, LLVM_ROOT, selectOpts)

        stateNew = interpretFIResults()

        agent.trainShortMemory(stateOld, action, reward, stateNew)

        agent.remember(stateOld, action, reward, stateNew)

        agent.n_iterations += 1
        agent.train_long_memory()

        if sdcRate < record
            record = sdcRate
            agent.model.save()

        print (f"Epoch {epoch}, sdcRate = {sdcRate}")


    print("All done!")

if __name__ == "__main__":
    main()
