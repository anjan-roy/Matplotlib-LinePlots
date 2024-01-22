s=input("Enter string:\n")
sub=input("Enter substring:\n")
i=s.find(sub)
if i==-1:
    print("Substring not found!")
while i!=-1:
    print("{} Present at index is: {}".format(sub,i))
    i=s.find(sub,i+len(sub),len(s))