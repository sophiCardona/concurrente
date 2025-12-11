import matplotlib.pyplot as plt

technologies = ['Python Multiprocessing\n(4 Procesos)', 'C + OpenMP\n(8 Hilos)']
times = [293.10, 114.79]

plt.figure(figsize=(8, 6))
bars = plt.bar(technologies, times, color=['#FFD700', '#005A9C']) 

plt.title('Comparación de Tiempo de Ejecución: Python vs C')
plt.ylabel('Tiempo Total (Segundos) ')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Poner el valor encima de la barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f"{yval:.2f} s", ha='center', fontweight='bold')

plt.savefig("comparacion_cpu.png")
plt.show()