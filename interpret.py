import re
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
import math

def remove_math(s):
    split = s.split("math.")
    final_str = ""
    for i in split:
        final_str += i
    return final_str

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

sample = "51n(pi/6)+51n(a)=1"
sample2 = "51n(pi/6)"
sample3 = "3+4"
sample4 = "a**2+64=100"

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
        equals[0] = remove_math(equals[0])
        equals[1] = remove_math(equals[1])
        print("this is 1 " + equals[0])
        print("this is 2 " + equals[1])
        answer = solveset(Eq(parse_expr(equals[0]),parse_expr(equals[1])),a,domain=S.Reals)
        if str(type(answer)) == "<class 'sympy.sets.sets.Union'>":
            components = answer.args
            answers = []
            for c in components:
                answers.append(str(c.args[0].args[1]).split(" ")[-1])
            return answers
        return answer
    else:
        if "math.math" in s:
            s = s.replace("math.math", "math")
        print("Final string is:" + s)
        return eval(s)

def interpret_old(s):
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
        equals[0] = remove_math(equals[0])
        equals[1] = remove_math(equals[1])
        print("this is 1 " + equals[0])
        print("this is 2 " + equals[1])
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
