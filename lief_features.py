import lief

if __name__=="__main__":
    file = './sample/beign_sample/KakaoTalk_Setup.exe'
    f=open(file, 'rb')
    #print(f.read())
    binary = lief.parse(file)

    bug = binary.optional_header
    #print(dir(bug))
    print(bug.dll_characteristics_lists)
    #print(dir(lief.PE.DataDirectory))
    #hahaha