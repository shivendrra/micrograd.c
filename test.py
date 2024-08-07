from micrograd import Value
import torch

# Micrograd operations
a = Value(2)
b = Value(3)

c = a + b
d = a * b
e = d.relu()
f = d.tanh()
g = d.sigmoid()
h = a - b
g.backward()

print('\nmicrograd values')
print(a.data, a.grad)
print(b.data, b.grad)
print(c.data, c.grad)
print(d.data, d.grad)
print(e.data, e.grad)
print(f.data, f.grad)

# PyTorch operations
a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)

c = a + b
d = a * b
e = d.relu()
f = d.tanh()
g = d.sigmoid()
h = a - b

c.retain_grad()
d.retain_grad()
e.retain_grad()
f.retain_grad()
g.backward()

print('\npytorch tensor')
print(a.data.item(), a.grad)
print(b.data.item(), b.grad)
print(c.data.item(), c.grad)
print(d.data.item(), d.grad)
print(e.data.item(), e.grad)
print(f.data.item(), f.grad)