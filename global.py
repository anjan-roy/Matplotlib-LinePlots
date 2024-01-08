#global variable in function

a=777
def f1():
    global a
    a=10
    print(a)
    a=999 #this remains a s global variable as it is already declared inside function
    print(a)
def f2():
    print(a)
f1()
f2()