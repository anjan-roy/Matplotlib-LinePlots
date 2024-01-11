#function to get even elements in a list
l=[0,1,2,3,4,5,6,7,8,9,10]

#lambda n:n%2==0

l1=list(filter(lambda n:n%2==0,l))
print(l1)
