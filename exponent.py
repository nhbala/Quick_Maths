from sklearn.cluster import KMeans
import numpy as np

sample = [(4,4,4,4),
(8,4,4,4),
(9,5,2,2),
(12,4,4,4),
(16,4,4,4),
(20,4,4,4),
(21,5,2,2),
]

sample2 = [(206,142,102,125),
(282,17,82,153),
(394,142,76,123),
(563,75,83,178),
(684,81,84,192)]

sample3 = [
[382, 423, 153, 189],
[577, 246, 66, 157],
[773, 425, 92, 160],
[1105, 352, 84, 215],
[1273, 228, 64, 152],
[1441, 416, 101, 115],
[1667, 311, 78, 256],
[1854, 368, 109, 122],
[2050, 293, 116, 240]
]

def expo(coor_lst):
    expo_lst = [0]
    for i in range(1,len(coor_lst)):
        lowBound = coor_lst[i][1] + coor_lst[i][3]
        midL = coor_lst[i-1][1] + (coor_lst[i-1][3]/2)
        midR = midL
        if i < len(coor_lst)-1:
            midR = coor_lst[i+1][1] + (coor_lst[i+1][3]/2)
        if lowBound < midL or lowBound < midR:
            expo_lst.append(1)
        else:
            expo_lst.append(0)
    return expo_lst
