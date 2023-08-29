import collections



class Solution:
    """
    @param buildings: A list of lists of integers
    @return: Find the outline of those buildings
    """
    def buildingOutline(self, buildings):
        # write your code here
        if not buildings: return []
        ans = []
        hash_table = {}
        N = len(buildings)
        index = []
        for i in range(N):
            start = buildings[i][0]
            end = buildings[i][1]
            span = end - start
            index.append(start)
            for j in range(buildings[i][0], start+span):
                char = str(j)
                if char not in hash_table:
                    hash_table[char] = buildings[i][2]
                elif hash_table[char] < buildings[i][2]:
                    hash_table[char] = buildings[i][2]
        #print(hash_table)
        #print(index)

        #get answer
        index = list(set(index))
        index.sort()
        #print(index)
        index = collections.deque(index)
        last_index = index[0]
        while index:
            i = index.popleft()
            if i < last_index:
                continue
            count = i
            start = count
            val = hash_table[str(i)]
            last_value = val
            count += 1
            while str(count) in hash_table:
                if hash_table[str(count)] == last_value:
                    last_value = hash_table[str(count)]
                    count += 1
                else:
                    index.appendleft(count)
                    break
            end = count
            ans.append([start, end, val])
            last_index = end
        print("ans", ans)
        return ans







P1 = Solution()
buildings = [
    [1, 3, 3],
    [2, 4, 4],
    [5, 6, 1]
]
buildings = [
    [1, 4, 3],
    [6, 9, 5]
]
buildings = [[1,10,3],[2,5,8],[7,9,8]]
buildings = [[4,67,187],[3,80,65],[49,77,117],[67,74,9],[6,42,92],[48,67,69],[10,13,58],[47,99,152],[66,99,53],[66,71,34],[27,63,2],[35,81,116],[47,49,10],[68,97,175],[20,33,53],[24,94,20],[74,77,155],[39,98,144],[52,89,84],[13,65,222],[24,41,75],[16,24,142],[40,95,4],[6,56,188],[1,38,219],[19,79,149],[50,61,174],[4,25,14],[4,46,225],[12,32,215],[57,76,47],[11,30,179],[88,99,99],[2,19,228],[16,57,114],[31,69,58],[12,61,198],[70,88,131],[7,37,42],[5,48,211],[2,64,106],[49,73,204],[76,88,26],[58,61,215],[39,51,125],[13,38,48],[74,99,145],[4,12,8],[12,33,161],[61,95,190],[16,19,196],[3,84,8],[5,36,118],[82,87,40],[8,44,212],[15,70,222],[16,25,176],[9,100,74],[38,78,99],[23,77,43],[45,89,229],[7,84,163],[48,72,1],[31,88,123],[35,62,190],[21,29,41],[37,97,81],[7,49,78],[83,84,132],[33,61,27],[18,45,1],[52,64,4],[58,98,57],[14,22,1],[9,85,200],[50,76,147],[54,70,201],[5,55,97],[9,42,125],[31,88,146]]
P1.buildingOutline(buildings)

