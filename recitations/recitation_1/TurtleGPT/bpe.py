
def get_workspace(words):
    workspace = {}
    for word in words:
        symbols = ' '.join(word) + ' æ'
        if not symbols in workspace:
            workspace[symbols] = 1
        else:
            workspace[symbols] += 1

    return workspace

def get_stats(workspace):
    pairs = {}
    for word, freq in workspace.items():
        symbols = word.split()
        for i in range(0,len(symbols)-1):
            pair = (symbols[i], symbols[i+1])
            if not pair in pairs:
                pairs[pair] = freq
            else:
                pairs[pair] += freq
    return pairs

def apply_merge(pair, workspace):
    new_workspace = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in workspace:
        new_word = word.replace(bigram, replacement)
        new_workspace[new_word] = workspace[word]
    return new_workspace

