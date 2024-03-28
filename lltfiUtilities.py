import os

def compileAndRun(filename, llfiRoot, llvmRoot):
    os.system('rm -rf ./llfi*')

    comp = "clang++ -w -emit-llvm -fno-unroll-loops -lstdc++ -fno-use-cxa-atexit -S *.cc"
    os.system(comp)

    link = f"{llvmRoot}llvm-link -o {filename}.ll -S *.ll"
    os.system(link)

    opt = f"{llvmRoot}opt \"{filename}.ll\" --disable-preinline -time-passes -S -o {filename}.ll"
    os.system(opt)

    instrument = f"{llfiRoot}/bin/instrument -lstdc++ --readable \"{filename}.ll\""
    os.system(instrument)

    profile = f"{llfiRoot}/bin/profile ./llfi/\"{filename}-profile.exe\" 10"
    os.system(profile)

    injection = f"{llfiRoot}/bin/injectfault ./llfi/\"{filename}-faultinjection.exe\" 10"
    os.system(injection)

def interpretFIResults():
    temp = "coming soon" 

if __name__ == "__main__":
    compileAndRun("test", ".", "~/llvm-project/build/bin/") 
    interpretFIResults()
