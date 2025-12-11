
# grafica de speedup version openMP con diferentes hilos 
import matplotlib.pyplot as plt

hilos = [1, 2, 4, 8, 12]
tiempos = [700.55, 701.82, 224.89, 114.79, 126.54]

# Calcular Speedup tomando el tiempo de 1 hilo como base
t_base = tiempos[0]
speedups = [t_base / t for t in tiempos]
ideal_speedup = hilos # Línea ideal

plt.figure(figsize=(10, 6))

# Gráfica Real
plt.plot(hilos, speedups, marker='o', linewidth=2, color='blue', label='Real (OpenMP)')
# Gráfica Ideal
plt.plot(hilos, ideal_speedup, '--', color='gray', label='Ideal (Lineal)')

plt.title('Análisis de Speedup')
plt.xlabel('Número de Hilos')
plt.ylabel('Speedup (X veces más rápido)')
plt.xticks(hilos)
plt.grid(True)
plt.legend()

# Etiquetas de valores
for i, txt in enumerate(speedups):
    plt.annotate(f"{txt:.2f}x", (hilos[i], speedups[i]), 
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.savefig("speedup_final.png")
plt.show()