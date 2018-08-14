# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import math

print("Hello World")


var1 = [1, 2, 3, 4]
var2 = True


# Create lists first and second
first = [11.25, 18.0, 20.0,9.50,9.50]
second = [10.75, 9.50]

# Paste together first and second: full
full = first + second

# Sort full in descending order: full_sorted
fully_sorted = sorted(full, reverse=False)

# Print out full_sorted
print(fully_sorted)
print("Maximum Value is :", max(fully_sorted))
print(len(fully_sorted))

mystring="hello world"
print(mystring.capitalize())
print(first.index(20.0))
print(first.count(9.50))
print(mystring.index("w"))
print(first.index(9.50))

first.append("My Element")

myint = 12
myresult = myint * math.pi
print(myresult)


count = 0
while (count < 9):
   print('The count is:', count)
   count = count + 1

print("Good bye!")


var = 1
while var == 1 :  # This constructs an infinite loop
   num = raw_input("Enter a number  :")
   print("You entered: ", num)

print"Good bye!"
