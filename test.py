from micrograd import Value

a = Value(2)
b = Value(3)

c = a+b
d = a*b
e = d.relu()
f = d.tanh()
g = d.sigmoid()
h = a - b
g.backward()

print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
print(g)