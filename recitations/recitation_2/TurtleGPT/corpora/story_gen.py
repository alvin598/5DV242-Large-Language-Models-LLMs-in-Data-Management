import random

rules = [
    ["S","NP VP"],
    ["Q","who does NP V # NP $"],
    ["R","if X1 hates X2 and X3 bites X2 then how does X1 feel about X3 # because X1 is an enemy of X2 and X3 is an enemy of X2 then X1 is a friend of X2 thus X1 loves X3 @ X1 loves X3 $"],
    ["R","if X1 bites X2 and X3 bites X2 then how does X1 feel about X3 # because X1 is an enemy of X2 and X3 is an enemy of X2 then X1 is a friend of X2 thus X1 loves X3 @ X1 loves X3 $"],
    ["R","if X1 loves X2 and X2 hates X3 then how does X1 feel about X3 # because X1 is a friend of X2 and X2 is an enemy of X3 then X1 is an enemy of X3 thus X1 hates X3 @ X1 hates X3 $"],
    ["R","if X1 loves X2 and X2 loves X3 then how does X1 feel about X3 # because X1 is a friend of X2 and X2 is an friend of X3 then X1 is a friend of X3 thus X1 loves X3 @ X1 loves X3 $"],

    ["NP","DET N"],
    ["NP","DET A N"],

    ["A","dirty"],
    ["A","clean"],
    ["A","rich"],
    ["A","poor"],
    ["A", "young"],
    ["A", "old"],

    ["DET","the"],
    ["DET","a"],

    ["N","dog"],
    ["N","cat"],
    ["N", "frog"],
    ["N", "rabbit"],

    ["VP","V NP"],

    ["V","hits"],
    ["V","bites"],
    ["V","knows"],
    ["V","loves"],
    ["V","hates"],
    ["V","admires"],
]

def random_sentence(sentence):
    while True:
        possible_rules = []
        for rule in rules:
            if sentence.find(' ' + rule[0] + ' ')>-1:
                possible_rules.append(rule)

        if not possible_rules:
            break
        rule = random.choice(possible_rules)
        sentence = sentence.replace(rule[0],rule[1],1)
    return sentence.strip() + ".\n"

"""
with open("stories.txt", "w") as my_file:
    for i in range(1,1000):
        my_file.write(f"{random_sentence(' S ')}")

with open("stories-sft.txt", "w") as my_file:
    for i in range(1,1000):
        my_file.write(f"{random_sentence(' Q ')}")
"""

with open("reasoning-sft.txt", "w") as my_file:
    for i in range(0,1000):
        line = f"{random_sentence(' R ')}"
        picks = ["dog", "cat", "frog", "rabbit"]
        random.shuffle(picks)
        line = line.replace("X1",picks[0])
        line = line.replace("X2", picks[1])
        line = line.replace("X3", picks[2])
        my_file.write(line)
