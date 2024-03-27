import os

def compileAndRun(filename, llfiRoot, llvmRoot):
    os.system('rm -rf ./llfi*')

    comp = "clang++ -w -emit-llvm -fno-unroll-loops -lstdc++ -fno-use-cxa-atexit -S *.cc"
    os.system(comp)

    link = llvmRoot + "llvm-link -o \"" + filename + ".ll\" -S *.ll"
    os.system(link)

    opt = llvmRoot + "opt " + "\"" + filename + ".ll\" --disable-preinline -time-passes -S -o \"" + filename + ".ll\""
    os.system(opt)

    instrument = llfiRoot + "/bin/instrument -lstdc++ --readable \"" + filename + ".ll\""
    os.system(instrument)

    profile = llfiRoot + "/bin/profile ./llfi/\"" + filename + "-profiling.exe\" 10"
    os.system(profile)

    injection = llfiRoot + "/bin/injectfault ./llfi/\"" + filename + "-faultinjection.exe\" 10"
    os.system(injection)

def interpretFIResults():
    temp = "coming soon" 

if __name__ == "__main__":
    compileAndRun()
    interpretFIResults()
