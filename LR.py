import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("data.csv")
#def loss(m,b,points):
#    total_loss=0
#    for i in range (len(points)):
#        x=points.iloc[i].studytime
#        y=points.iloc[i].score
#        total_loss+= (y - (m*x + b))**2
#    total_loss / float(len(points))

def gradient(m,b,points,L):
    m_gradient=0
    b_gradient=0

    n=len(points)

    for i in range(n):
        x=points.iloc[i].studytime
        y=points.iloc[i].score
        m_gradient += -(2/n) * x * (y - (m*x + b))
        b_gradient += -(2/n) * (y - (m*x + b))

    m = m - (L * m_gradient)
    b = b - (L * b_gradient)
    return m, b

m=0
b=0
L=0.0001
epochs=1000

for i in range(epochs):
    if i%50==0:
        print(f'Epoch {i}: m={m}, b={b}')
    m, b = gradient(m, b, data, L)

print(m,b)
plt.scatter(data.studytime, data.score, color='blue')
plt.plot(list(range(0, 10)), [m*x + b for x in range(0, 10)], color='red')
plt.show
    