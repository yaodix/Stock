


a = {2:["1",2,3,4], 4:["6",3,7,5]}
ratio_sum_sorted = sorted(a.items(),  key = lambda x:(x[1][1]+x[1][3]), reverse = True)
for  val in ratio_sum_sorted:
  print(f"code {val[0]}, break 1 {val[1][0]}, data {val[1][1]}, break 1 {val[1][2]}, data {val[1][3]}")

  