from typing import List

class Solution:
    def fl(self, nums: List[int], k:int) -> int:   
        
        lid = 0
        curr_sum = 0
        out = 0
        for rid in range(len(nums)):
            # add els from right
            curr_sum += nums[rid]
            # remove els from left if sum constraint broken
            while curr_sum > k:
                curr_sum -= nums[lid]
                lid += 1
            # track longest window so far
            out = max(out, rid - lid + 1)
            print(out)
          
            
        return out
    
    def fsl(self, s: str) -> int:   
        
        lid = 0
        curr_sum = 0
        out = 0
        for rid in range(len(s)):
            # add els from right
            if s[rid] == "0":
                curr_sum += 1
            # remove els from left if sum constraint broken
            while curr_sum > 1:
                if s[lid] == "0":
                    curr_sum -= 1
                lid += 1
            # track longest window so far
            out = max(out, rid - lid + 1)
            print(out)
          
            
        return out
    
    def lsa(self, nums: List[int], k:int) -> int:   
        
        # first window
        curr_sum = 0
        for i in range(k):
            curr_sum += nums[i]
        
        out = curr_sum
        for rid in range(k, len(nums)):
            # add els from right, remove from left
            lid = rid - k
            curr_sum += nums[rid] - nums[lid]
            # track longest sum so far
            out = max(out, curr_sum)
            print(out)
          
            
        return out    
    
    
    
      
nums = [3, 1, 2, 7, 4, 2, 1, 1, 5]
asol = Solution()
out = asol.fl(nums, 8)

nums = "1101100111"
asol = Solution()
out = asol.fsl(nums)

nums = [3, -1, 4, 12, -8, 5, 6]
asol = Solution()
out = asol.lsa(nums, 4)

class Solutions:
    def getAverages(self, nums: List[int], k: int) -> List[int]:
        
        # n = len(nums)
        # csum =[nums[0]]
        # for i in range(1,n):
        #     csum.append(csum[-1]+nums[i])
        
        # # print(csum)
        # avgk = []
        # csumk = 0
        # win_len = 2*k + 1 # i+k - (i-k) + 1
        # for i in range(n):
        #     if ((i-k) < 0) or ((i+k) > (n-1)):
        #         avgk.append(-1)
        #     else:
        #         csumk = csum[i+k] - csum[i-k] + nums[i-k]
        #         # print(f"{csumk},{k},{csumk//win_len}")
                
        #         avgk.append((csumk)//win_len)
                
        # return avgk

        
        n = len(nums)
        avgk = [-1]*n
        win_len = 2*k + 1 # i+k - (i-k) + 1
        wsumk = sum(nums[:win_len])
        # print(avgk[k])
        avgk[k] = wsumk//win_len

        for j in range(win_len,n):
            # if j-k>n-1:
            #     continue
            # else:
            wsumk += nums[j] - nums[j-win_len]
            avgk[j-k] = wsumk//win_len

        return avgk
    
    
asol = Solutions()
out = asol.getAverages([7,4,3,9,1,8,5,2,6], 3)
print(out)
