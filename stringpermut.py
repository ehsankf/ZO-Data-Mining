def permutations_without_dups2(s):
   if len(s)==1:
      return [s]
   permutation = list()
   for c in s:
    s_p = s.copy()
    s_p.remove(c)
    for i, permut in enumerate(permutations_without_dups2(s_p)):
      permut_cp = permut.copy()
      print(permut_cp)
      permut_cp.insert(0, c)
      permutation.append(permut_cp)

   return permutation


permutat = permutations_without_dups2(list("abc"))

print(permutat)

