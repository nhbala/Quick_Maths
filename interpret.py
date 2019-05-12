import re
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
import math

#figure out if trig functions are involved in equation
def trig(s):
    print("At trig, s is:"+str(type(s)))
    record = []
    weird = False
    for i in range(len(s)):
        curr = s[i]
        if curr == "n":
            #sin
            if s[i-2] == "5" or s[i-2] == "s":
                s = s[:i-2] + "si" + s[i:]
                #see if output will be in lambda equation format
                temp = i+1
                while s[temp] != ')':
                    if s[temp] == 'a':
                        weird = True
                    temp = temp + 1
                record.append(("sin",i-2))
            else:
                #tan
                s = s[:i-2] + "ta" + s[i:]
                #see if output will be in lambda equation format
                temp = i+1
                while s[temp] != ')':
                    if s[temp] == 'a':
                        weird = True
                    temp = temp + 1
                record.append(("tan",i-2))
        #cos
        elif curr == "c":
            s = s[:i+1] + "os" + s[i+3:]
            #see if output will be in lambda equation format
            temp = i+1
            while s[temp] != ')':
                if s[temp] == 'a':
                    weird = True
                temp = temp + 1
            record.append(("cos",i))
    return s,record, weird

def poly(s):
    for i in range(len(s)):
        curr = s[i]
        if curr in ["a","t","s","c","(","m"] and i != 0:
            if s[i-1].isdigit():
                s = s[:i] + "*" + s[i:]
    return s

def handleE(s):
    for i in range(len(s)):
        curr = s[i]
        if curr == "e":
            s = s[:i] + "math.e" + s[i+1:]
    return s

def handlePi(s):
    for i in range(len(s)):
        curr = s[i]
        if curr == "p":
            if s[i-1] != "." and i > 0:
                s = s[:i] + "math.pi" + s[i+2:]
    return s

def interpret(s):
    print("At interepret, s is:"+str(type(s)))
    s,record, weird = trig(s)
    s = handleE(s)
    s = handlePi(s)
    s = poly(s)
    equals = s.split("=")
    if len(equals) > 2:
        print("Goofed")
    elif len(equals) == 2:
        a = Symbol("a")
        answer = solveset(Eq(parse_expr(equals[0]),parse_expr(equals[1])),a,domain=S.Reals)
        print(str(answer))
        if weird:
            answer2 = ""
            sets = str(answer).split('U')
            print("Sets are: "+str(sets))
            for set in sets:
                parts = set.split(',')
                print("parts are: "+str(parts))
                answer2 += '{' + parts[1][1:-1] + '} U '
            answer2 = answer2[:-3]
            return answer2
        return answer
    else:
        print("Final string is:" + s)
        return eval(s)

def unwrap(S):
    for s in S:
        print(str(s))
