s=input("Enter String:\n")

if s.isalpha():
    print("Given string is alphabets only")
    if s.islower():
        print("Given string is lower case")
    elif s.isupper():
        print("Given string are upper case")
elif s.isdigit():
    print("Given string is digits only")
elif s.istitle():
    print("Given string are of title type")
elif s.isspace():
    print("Only space")