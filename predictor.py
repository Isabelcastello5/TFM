import numpy as np
from scipy.optimize import curve_fit

# ==================================================
# MODEL EXPONENCIAL PER ESTIMAR EL MÒDUL D'ELASTICITAT (E) A PARTIR DE FD
# ==================================================

# Dades experimentals d'entrada
# FD: dimensió fractal mitjana per mostra
# E: mòdul d’elasticitat aparent mesurat (en MPa)
valors_fd = np.array([2.832, 2.991, 2.662, 2.811, 2.636, 2.701, 2.665])
valors_e  = np.array([2.459, 0.750, 12.209, 3.776, 18.003, 11.046, 11.604])

# Definició del model exponencial
# E = E0 * exp(-alpha * FD)
def model_exponencial(fd, e0, alpha):
    return e0 * np.exp(-alpha * fd)

# Ajust dels paràmetres del model a les dades experimentals 
paràmetres_optimitzats, _ = curve_fit(model_exponencial, valors_fd, valors_e, p0=[20, 1])
e0_opt, alpha_opt = paràmetres_optimitzats

# Prediccions per a les dades d'entrenament
e_predites = model_exponencial(valors_fd, e0_opt, alpha_opt)

# Càlcul del coeficient de determinació (R²)
residus = valors_e - e_predites
suma_residus = np.sum(residus**2)
suma_total = np.sum((valors_e - np.mean(valors_e))**2)
r_quadrat = 1 - (suma_residus / suma_total)

# Mostrar resultats per consola
print(f"\n📈 Resultats del model exponencial ajustat:")
print(f"  E₀ (paràmetre inicial)     = {e0_opt:.4f}")
print(f"  α (coeficient de decaïment) = {alpha_opt:.4f}")
print(f"  R² (qualitat de l’ajust)    = {r_quadrat:.4f}")

# Predicció interactiva des de terminal
try:
    entrada_usuari = input("\n✏️ Introdueix un valor de FD per predir E (mòdul d’elasticitat): ")
    fd_nou = float(entrada_usuari)
    e_nou = model_exponencial(fd_nou, e0_opt, alpha_opt)
    print(f"\n🔎 Per FD = {fd_nou:.4f}, la predicció de E és: {e_nou:.4f} MPa")
except ValueError:
    print("⚠️ Entrada no vàlida. Si us plau, introdueix un nombre decimal vàlid.")
