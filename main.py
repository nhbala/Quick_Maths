from cv import cv_imgs
from trying import get_rep
from exponent import expo
from interpret import interpret




if __name__ == "__main__":
    curr_lst, exp_lst = cv_imgs('pictures/handwritten.JPG')
    #1 is exp, 0 != exp
    is_exp = expo(exp_lst)
    final_array = []
    for item in curr_lst:
        curr_val = get_rep(item)
        final_array.append(curr_val)

    final_str = ""
    flag = False
    print(final_array)
    print(is_exp)
    for index in range(len(is_exp)):
        if is_exp[index] == 0:
            if flag == False:
                final_str += final_array[index]
            else:
                final_str += (")" + final_array[index])
                flag = False
        else:
            if flag == False:
                final_str += "**(" + final_array[index]
                if index  == len(is_exp) - 1:
                    final_str += ")"
                else:
                    flag = True
            else:
                final_str += final_array[index]

    print('Your Equation is: ' + final_str)
    eval = interpret(final_str)
    print(eval)
