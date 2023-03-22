import math
import numpy as np

class Results():
    def __init__(self):
        var1 = 0

    def getHitRatio(self, 
                    recommend_list: list,
                    gt_item: int) -> bool:
        if gt_item in recommend_list:
            return 1
        else:
            return 0

    def getNDCG(self, 
                recommend_list: list,
                gt_item: int) -> float:
        idx = np.where(recommend_list == gt_item)[0]
        if len(idx) > 0:
            return math.log(2)/math.log(idx+2)
        else:
            return 0
    
    def coverage(self, urecol, total_items):
        return len(set(np.hstack(urecol))) / total_items *100
