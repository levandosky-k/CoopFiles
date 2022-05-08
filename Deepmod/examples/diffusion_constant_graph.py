import matplotlib.pyplot as plt
  
x = []
y = []
for line in open('diffusion_graph.txt', 'r'):
    lines = [i for i in line.split()]
    x.append(lines[0])
    y.append(int(lines[1]))
      
plt.title("Diffusion Constants vs Correctness")
plt.xlabel('Diffusion Constant')
plt.ylabel('Correct')
plt.yticks(y)
plt.plot(x, y, marker = 'o', c = 'g')
  
plt.show()
